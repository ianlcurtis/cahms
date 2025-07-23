"""
Groundedness Evaluator for CAHMS Assessment Reports
Uses Azure AI Foundry's evaluation SDK to assess whether generated reports
are grounded in the provided source documents.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Azure AI Evaluation imports
try:
    from azure.ai.evaluation import GroundednessEvaluator, RelevanceEvaluator, AzureOpenAIModelConfiguration
    from azure.identity import DefaultAzureCredential
    EVALUATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Azure AI Evaluation SDK not available: {e}")
    EVALUATION_AVAILABLE = False

# Import CAHMS modules
try:
    from azure_llm_client_sk import AzureLLMClientSemanticKernel
    from document_extractor import DocumentExtractor, process_assessment_documents
    from assessment_prompt import AssessmentPromptGenerator
    CAHMS_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"CAHMS modules not available: {e}")
    CAHMS_MODULES_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationCase:
    """Represents a single evaluation test case"""
    case_id: str
    documents: Dict[str, bytes]  # filename -> content
    metadata: Optional[Dict[str, Any]] = None
    expected_output: Optional[str] = None


@dataclass
class EvaluationResult:
    """Results from groundedness evaluation"""
    case_id: str
    groundedness_score: float
    relevance_score: float
    response_length: int
    evaluation_timestamp: datetime
    detailed_feedback: Dict[str, Any]
    passed_threshold: bool
    generation_time_seconds: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['evaluation_timestamp'] = self.evaluation_timestamp.isoformat()
        return result


class CAHMSGroundednessEvaluator:
    """Evaluates groundedness of CAHMS assessment reports"""
    
    def __init__(self, config_path: str = "configs/evaluation_config.json"):
        self.config = self._load_config(config_path)
        
        # Check if all required components are available
        if not EVALUATION_AVAILABLE:
            logger.error("Azure AI Evaluation SDK not available. Please install: pip install azure-ai-evaluation")
            self.configured = False
            return
            
        if not CAHMS_MODULES_AVAILABLE:
            logger.error("CAHMS modules not available. Please check your src directory.")
            self.configured = False
            return
        
        # Initialize CAHMS components
        self.llm_client = AzureLLMClientSemanticKernel()
        self.document_extractor = DocumentExtractor()
        self.prompt_generator = AssessmentPromptGenerator()
        
        # Initialize Azure AI evaluators
        self.configured = self._init_evaluators()
        
        # Evaluation thresholds
        self.groundedness_threshold = self.config.get("groundedness_threshold", 3.0)
        self.relevance_threshold = self.config.get("relevance_threshold", 3.0)
        self.pass_rate_threshold = self.config.get("pass_rate_threshold", 0.8)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load evaluation configuration"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Substitute environment variables
            for key, value in config.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    config[key] = os.getenv(env_var, value)
                    
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                "groundedness_threshold": 3.0,
                "relevance_threshold": 3.0,
                "azure_openai_endpoint": os.getenv("LLM_ENDPOINT"),
                "azure_openai_api_key": os.getenv("LLM_API_KEY"),
                "evaluation_model": "gpt-4",
                "pass_rate_threshold": 0.8
            }
    
    def _init_evaluators(self) -> bool:
        """Initialize Azure AI Foundry evaluators"""
        try:
            # Configure the model for evaluation
            model_config = AzureOpenAIModelConfiguration(
                azure_endpoint=self.config["azure_openai_endpoint"],
                api_key=self.config["azure_openai_api_key"],
                azure_deployment=self.config.get("evaluation_model", "gpt-4"),
                api_version="2024-02-15-preview"
            )
            
            # Initialize evaluators
            self.groundedness_evaluator = GroundednessEvaluator(model_config)
            self.relevance_evaluator = RelevanceEvaluator(model_config)
            
            logger.info("Azure AI evaluators initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize evaluators: {e}")
            return False
    
    def is_configured(self) -> bool:
        """Check if evaluator is properly configured"""
        return self.configured and self.llm_client.is_configured()
    
    async def load_test_cases(self, test_documents_path: str) -> List[EvaluationCase]:
        """Load test cases from the test documents directory"""
        test_cases = []
        
        # Scan for test case directories
        test_path = Path(test_documents_path)
        if not test_path.exists():
            logger.error(f"Test documents path {test_documents_path} does not exist")
            return test_cases
            
        for case_dir in test_path.iterdir():
            if not case_dir.is_dir():
                continue
            
            case_id = case_dir.name
            logger.info(f"Loading test case: {case_id}")
            
            # Load metadata
            metadata_path = case_dir / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Load documents
            documents = {}
            for file_path in case_dir.iterdir():
                if file_path.suffix.lower() in ['.pdf', '.docx', '.txt'] and file_path.name != 'metadata.json':
                    with open(file_path, 'rb') as f:
                        documents[file_path.name] = f.read()
            
            if not documents:
                logger.warning(f"No documents found in case {case_id}")
                continue
            
            # Load expected output if available
            expected_output = None
            expected_path = Path("expected_outputs") / f"{case_id}_expected.txt"
            if expected_path.exists():
                with open(expected_path, 'r') as f:
                    expected_output = f.read()
            
            test_cases.append(EvaluationCase(
                case_id=case_id,
                documents=documents,
                metadata=metadata,
                expected_output=expected_output
            ))
        
        logger.info(f"Loaded {len(test_cases)} test cases")
        return test_cases
    
    async def generate_response_for_case(self, test_case: EvaluationCase) -> tuple[str, float]:
        """Generate a CAHMS assessment response for a test case"""
        start_time = datetime.now()
        
        # Create mock file objects for processing
        class MockFile:
            def __init__(self, name: str, content: bytes):
                self.name = name
                self._content = content
            
            def read(self) -> bytes:
                return self._content
        
        # Map document filenames to expected keys
        document_mapping = {
            "form_s.pdf": "form_s",
            "form_s.txt": "form_s",
            "form_h.pdf": "form_h", 
            "form_h.txt": "form_h",
            "form_a.pdf": "form_a",
            "form_a.txt": "form_a",
            "cahms_initial.pdf": "cahms_initial",
            "cahms_initial.txt": "cahms_initial",
            "neuro_dev_history.pdf": "neuro_dev_history",
            "neuro_dev_history.txt": "neuro_dev_history",
            "formulation_document.pdf": "formulation_document",
            "formulation_document.txt": "formulation_document",
            "school_observation.pdf": "school_observation",
            "school_observation.txt": "school_observation",
            "supporting_information.pdf": "supporting_information",
            "supporting_information.txt": "supporting_information"
        }
        
        uploaded_files_dict = {}
        for filename, content in test_case.documents.items():
            key = document_mapping.get(filename)
            if key:
                uploaded_files_dict[key] = MockFile(filename, content)
        
        if not uploaded_files_dict:
            raise Exception(f"No mappable documents found for case {test_case.case_id}")
        
        # Process documents using existing CAHMS workflow
        session_id = f"eval_{test_case.case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Define mandatory uploads based on available documents
        mandatory_uploads = {key: True for key in uploaded_files_dict.keys()}
        # At minimum, require form_s
        mandatory_uploads.update({
            "form_h": False, "form_a": False, "cahms_initial": False,
            "neuro_dev_history": False, "formulation_document": False, 
            "school_observation": False, "supporting_information": False
        })
        
        assessment_request = await process_assessment_documents(
            uploaded_files_dict, 
            session_id, 
            mandatory_uploads
        )
        
        if not assessment_request.documents:
            raise Exception(f"No documents processed for case {test_case.case_id}")
        
        # Generate prompt and system message
        prompt = self.prompt_generator.create_assessment_prompt(assessment_request.documents)
        system_message = self.prompt_generator.create_system_message()
        
        # Create request object
        class PromptRequest:
            def __init__(self, prompt: str, system_message: str, documents: List, session_id: str):
                self.prompt = prompt
                self.system_message = system_message
                self.documents = documents
                self.session_id = session_id
                self.request_timestamp = datetime.now()
        
        prompt_request = PromptRequest(prompt, system_message, assessment_request.documents, session_id)
        
        # Generate response
        result = await self.llm_client.generate_response(prompt_request)
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        if result["success"]:
            return result["report"], generation_time
        else:
            raise Exception(f"Failed to generate response: {result['error']}")
    
    async def evaluate_groundedness(self, 
                                  response: str, 
                                  source_documents: Dict[str, bytes],
                                  test_case: EvaluationCase,
                                  generation_time: float) -> EvaluationResult:
        """Evaluate groundedness of a generated response"""
        
        try:
            # Extract text content from documents for context
            context_parts = []
            for filename, content in source_documents.items():
                try:
                    # Simple text extraction for evaluation context
                    if filename.endswith('.txt'):
                        text_content = content.decode('utf-8')
                    else:
                        # For PDF/DOCX, use a simplified approach
                        text_content = f"[Content from {filename}]"
                    
                    context_parts.append(f"Document: {filename}\nContent: {text_content}")
                except Exception as e:
                    logger.warning(f"Could not extract text from {filename}: {e}")
                    context_parts.append(f"Document: {filename}\n[Unable to extract content]")
            
            context = "\n\n".join(context_parts)
            
            # Evaluate groundedness
            groundedness_result = self.groundedness_evaluator(
                query="Generate a comprehensive neurodevelopmental assessment report based on the provided documents",
                response=response,
                context=context
            )
            
            # Evaluate relevance
            relevance_result = self.relevance_evaluator(
                query="Generate a comprehensive neurodevelopmental assessment report based on the provided documents", 
                response=response,
                context=context
            )
            
            # Extract scores (Azure AI Evaluation returns scores 1-5)
            groundedness_score = float(groundedness_result.get("groundedness", 0.0))
            relevance_score = float(relevance_result.get("relevance", 0.0))
            
            # Determine if thresholds are met
            passed_threshold = (
                groundedness_score >= self.groundedness_threshold and
                relevance_score >= self.relevance_threshold
            )
            
            return EvaluationResult(
                case_id=test_case.case_id,
                groundedness_score=groundedness_score,
                relevance_score=relevance_score,
                response_length=len(response),
                evaluation_timestamp=datetime.now(),
                detailed_feedback={
                    "groundedness_details": groundedness_result,
                    "relevance_details": relevance_result,
                    "thresholds": {
                        "groundedness_threshold": self.groundedness_threshold,
                        "relevance_threshold": self.relevance_threshold
                    },
                    "context_length": len(context),
                    "document_count": len(source_documents)
                },
                passed_threshold=passed_threshold,
                generation_time_seconds=generation_time
            )
            
        except Exception as e:
            logger.error(f"Error evaluating case {test_case.case_id}: {e}")
            return EvaluationResult(
                case_id=test_case.case_id,
                groundedness_score=0.0,
                relevance_score=0.0,
                response_length=len(response) if response else 0,
                evaluation_timestamp=datetime.now(),
                detailed_feedback={"error": str(e)},
                passed_threshold=False,
                generation_time_seconds=generation_time,
                error_message=str(e)
            )
    
    async def run_evaluation_suite(self, test_documents_path: str) -> List[EvaluationResult]:
        """Run the complete evaluation suite"""
        logger.info("Starting CAHMS groundedness evaluation suite")
        
        if not self.is_configured():
            logger.error("Evaluator not properly configured")
            return []
        
        # Load test cases
        test_cases = await self.load_test_cases(test_documents_path)
        
        if not test_cases:
            logger.error("No test cases found")
            return []
        
        results = []
        for test_case in test_cases:
            logger.info(f"Evaluating case: {test_case.case_id}")
            
            try:
                # Generate response
                response, generation_time = await self.generate_response_for_case(test_case)
                
                # Evaluate groundedness
                result = await self.evaluate_groundedness(
                    response, test_case.documents, test_case, generation_time
                )
                
                results.append(result)
                
                status = "✅ PASSED" if result.passed_threshold else "❌ FAILED"
                logger.info(f"Case {test_case.case_id}: "
                          f"Groundedness={result.groundedness_score:.2f}, "
                          f"Relevance={result.relevance_score:.2f}, "
                          f"Time={result.generation_time_seconds:.1f}s, "
                          f"Status={status}")
                
            except Exception as e:
                logger.error(f"Error evaluating case {test_case.case_id}: {e}")
                # Create failed result
                results.append(EvaluationResult(
                    case_id=test_case.case_id,
                    groundedness_score=0.0,
                    relevance_score=0.0,
                    response_length=0,
                    evaluation_timestamp=datetime.now(),
                    detailed_feedback={"error": str(e)},
                    passed_threshold=False,
                    generation_time_seconds=0.0,
                    error_message=str(e)
                ))
        
        return results
    
    def generate_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate evaluation report"""
        total_cases = len(results)
        passed_cases = sum(1 for r in results if r.passed_threshold)
        failed_cases = [r for r in results if not r.passed_threshold]
        
        avg_groundedness = sum(r.groundedness_score for r in results) / total_cases if total_cases > 0 else 0
        avg_relevance = sum(r.relevance_score for r in results) / total_cases if total_cases > 0 else 0
        avg_generation_time = sum(r.generation_time_seconds for r in results) / total_cases if total_cases > 0 else 0
        
        pass_rate = passed_cases / total_cases if total_cases > 0 else 0
        
        return {
            "evaluation_summary": {
                "total_cases": total_cases,
                "passed_cases": passed_cases,
                "failed_cases": len(failed_cases),
                "pass_rate": pass_rate,
                "passed_threshold": pass_rate >= self.pass_rate_threshold,
                "avg_groundedness_score": avg_groundedness,
                "avg_relevance_score": avg_relevance,
                "avg_generation_time_seconds": avg_generation_time,
                "evaluation_timestamp": datetime.now().isoformat(),
                "thresholds": {
                    "groundedness_threshold": self.groundedness_threshold,
                    "relevance_threshold": self.relevance_threshold,
                    "pass_rate_threshold": self.pass_rate_threshold
                }
            },
            "case_results": [result.to_dict() for result in results],
            "failed_cases": [
                {
                    "case_id": r.case_id,
                    "groundedness_score": r.groundedness_score,
                    "relevance_score": r.relevance_score,
                    "error_message": r.error_message
                }
                for r in failed_cases
            ]
        }


async def main():
    """Main evaluation runner"""
    try:
        evaluator = CAHMSGroundednessEvaluator()
        
        if not evaluator.is_configured():
            logger.error("Evaluator not configured properly. Please check your environment variables and configuration.")
            exit(1)
        
        results = await evaluator.run_evaluation_suite("test_documents")
        
        if not results:
            logger.error("No evaluation results generated")
            exit(1)
        
        # Generate and save report
        report = evaluator.generate_report(results)
        
        # Save detailed results
        with open("evaluation_results.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        summary = report["evaluation_summary"]
        print(f"\n=== CAHMS Groundedness Evaluation Results ===")
        print(f"Total Cases: {summary['total_cases']}")
        print(f"Passed Cases: {summary['passed_cases']}")
        print(f"Failed Cases: {summary['failed_cases']}")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")
        print(f"Average Groundedness Score: {summary['avg_groundedness_score']:.2f}/5.0")
        print(f"Average Relevance Score: {summary['avg_relevance_score']:.2f}/5.0")
        print(f"Average Generation Time: {summary['avg_generation_time_seconds']:.1f}s")
        
        # Show failed cases
        if report["failed_cases"]:
            print(f"\n❌ Failed Cases:")
            for failed_case in report["failed_cases"]:
                print(f"  - {failed_case['case_id']}: G={failed_case['groundedness_score']:.2f}, R={failed_case['relevance_score']:.2f}")
                if failed_case.get('error_message'):
                    print(f"    Error: {failed_case['error_message']}")
        
        # Exit with appropriate code for CI/CD
        if summary['passed_threshold']:
            print(f"\n✅ Evaluation PASSED - pass rate {summary['pass_rate']:.1%} meets threshold {summary['thresholds']['pass_rate_threshold']:.1%}")
            exit(0)
        else:
            print(f"\n❌ Evaluation FAILED - pass rate {summary['pass_rate']:.1%} below threshold {summary['thresholds']['pass_rate_threshold']:.1%}")
            exit(1)
            
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        print(f"\n❌ Evaluation FAILED with error: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
