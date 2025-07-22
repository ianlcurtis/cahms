"""
Semantic Kernel Azure LLM Client for CAHMS Neurodevelopmental Assessment
Simplified version focused on core functionality.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Semantic Kernel imports with error handling
try:
    import semantic_kernel as sk
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
    from semantic_kernel.core_plugins.text_plugin import TextPlugin
    from semantic_kernel.functions.kernel_function_decorator import kernel_function
    from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
    from semantic_kernel.contents.chat_history import ChatHistory
    from semantic_kernel import Kernel
    SK_AVAILABLE = True
    print(f"✅ Semantic Kernel v{getattr(sk, '__version__', 'unknown')} imported")
except ImportError as e:
    print(f"❌ Semantic Kernel not available: {e}")
    SK_AVAILABLE = False
    sk = AzureChatCompletion = TextPlugin = kernel_function = None
    PromptExecutionSettings = ChatHistory = Kernel = None

# Prompty imports with error handling
try:
    from prompty import load, prepare, execute
    PROMPTY_AVAILABLE = True
    print("✅ Prompty imported")
except ImportError:
    print("❌ Prompty not available")
    PROMPTY_AVAILABLE = False
    load = prepare = execute = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureLLMClientSemanticKernel:
    """Semantic Kernel-based client for Azure LLM services"""
    
    def __init__(self):
        """Initialize the LLM client"""
        self.endpoint = os.getenv("LLM_ENDPOINT")
        self.api_key = os.getenv("LLM_API_KEY")
        self.model_name = os.getenv("LLM_MODEL_NAME", "gpt-4")
        
        self.kernel = None
        self.chat_service = None
        self._initialize_kernel()
        
        # Initialize helper components
        self._initialize_helpers()
    
    def _initialize_helpers(self):
        """Initialize helper modules"""
        try:
            from document_extractor import DocumentExtractor
            from assessment_prompt import AssessmentPromptGenerator
            self.document_extractor = DocumentExtractor()
            self.prompt_generator = AssessmentPromptGenerator()
        except ImportError:
            logger.warning("Helper modules not available. Limited functionality.")
            self.document_extractor = None
            self.prompt_generator = None
    
    def _initialize_kernel(self):
        """Initialize Semantic Kernel with Azure OpenAI"""
        if not all([self.endpoint, self.api_key, SK_AVAILABLE]):
            logger.warning("Missing credentials or Semantic Kernel unavailable")
            return
        
        try:
            self.kernel = Kernel()
            
            # Add Azure OpenAI chat service
            self.chat_service = AzureChatCompletion(
                service_id="azure_openai_chat",
                deployment_name=self.model_name,
                endpoint=self.endpoint,
                api_key=self.api_key,
                api_version="2024-02-15-preview"
            )
            self.kernel.add_service(self.chat_service)
            
            # Add plugins
            self._add_plugins()
            
            logger.info("✅ Semantic Kernel initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Semantic Kernel: {e}")
            self.kernel = None
    
    def _add_plugins(self):
        """Add plugins to kernel"""
        # Add text plugin
        try:
            self.kernel.add_plugin(TextPlugin(), plugin_name="text")
        except Exception as e:
            logger.warning(f"Could not add TextPlugin: {e}")
        
        # Add custom assessment plugin
        self._add_assessment_plugin()
        
        # Add Prompty plugin if available
        if PROMPTY_AVAILABLE:
            self._add_prompty_plugin()
    
    def _add_assessment_plugin(self):
        """Add custom assessment plugin"""
        if not self.kernel:
            return
        
        class AssessmentPlugin:
            def __init__(self, chat_service):
                self.chat_service = chat_service
            
            @kernel_function(description="Analyze assessment documents", name="analyze_documents")
            async def analyze_documents(self, documents: str) -> str:
                prompt = f"""
                Analyze these neurodevelopmental assessment documents:
                {documents}
                
                Provide:
                1. Document count and types
                2. Key concerns identified
                3. Strengths noted
                4. Recommendations
                """
                
                return await self._execute_prompt(prompt)
            
            @kernel_function(description="Validate document quality", name="validate_documents")
            def validate_documents(self, documents: str) -> str:
                doc_sections = documents.split('---DOCUMENT---')
                required_forms = ['Form S', 'Form H', 'Form A']
                found_forms = [form for form in required_forms if any(form in doc for doc in doc_sections)]
                
                return f"Found {len(found_forms)} standard forms: {found_forms}" if found_forms else "No standard forms detected"
            
            async def _execute_prompt(self, prompt: str) -> str:
                try:
                    chat_history = ChatHistory()
                    chat_history.add_user_message(prompt)
                    
                    result = await self.chat_service.get_chat_message_content(
                        chat_history=chat_history,
                        settings=PromptExecutionSettings(max_tokens=1000, temperature=0.2)
                    )
                    return str(result.content) if hasattr(result, 'content') else str(result)
                except Exception as e:
                    return f"Analysis failed: {e}"
        
        assessment_plugin = AssessmentPlugin(self.chat_service)
        self.kernel.add_plugin(assessment_plugin, plugin_name="assessment")
        logger.info("✅ Added AssessmentPlugin")
    
    def _add_prompty_plugin(self):
        """Add Prompty-based plugin"""
        if not self.kernel:
            return
        
        try:
            # Load prompty template
            prompty_file = os.path.join(os.path.dirname(__file__), "assessment_report.prompty")
            if not os.path.exists(prompty_file):
                logger.warning("Prompty template not found")
                return
            
            self.prompty_template = load(prompty_file)
            
            # Configure Prompty execution environment
            # Use environment variables or configuration instead of executor
            try:
                # Set up environment for Prompty execution
                os.environ.setdefault("AZURE_OPENAI_ENDPOINT", self.endpoint)
                os.environ.setdefault("AZURE_OPENAI_API_KEY", self.api_key)
                os.environ.setdefault("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
                os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT_NAME", self.model_name)
                
                logger.info("✅ Prompty environment configured")
                
            except Exception as config_error:
                logger.warning(f"Could not configure Prompty environment: {config_error}")
            
            class PromptyPlugin:
                def __init__(self, template, chat_service):
                    self.template = template
                    self.chat_service = chat_service
                
                @kernel_function(description="Generate assessment report", name="generate_report")
                async def generate_report(self, documents: str) -> str:
                    try:
                        if self.template:
                            # Use Prompty with environment-based configuration
                            # Pass variables as a dictionary to prepare()
                            prepared = prepare(self.template, {"documents": documents})
                            
                            # Execute without explicit model parameter (use environment config)
                            try:
                                result = await execute(prepared)
                                return str(result)
                            except Exception as exec_error:
                                logger.warning(f"Prompty execution failed: {exec_error}")
                                # Fall back to traditional generation
                                return await self._fallback_generation(documents)
                        else:
                            return await self._fallback_generation(documents)
                    except Exception as e:
                        logger.error(f"Prompty execution failed: {e}")
                        return await self._fallback_generation(documents)
                
                async def _fallback_generation(self, documents: str) -> str:
                    prompt = f"Generate a neurodevelopmental assessment report based on: {documents}"
                    chat_history = ChatHistory()
                    chat_history.add_user_message(prompt)
                    
                    result = await self.chat_service.get_chat_message_content(
                        chat_history=chat_history,
                        settings=PromptExecutionSettings(max_tokens=4000, temperature=0.3)
                    )
                    return str(result.content) if hasattr(result, 'content') else str(result)
        
            prompty_plugin = PromptyPlugin(self.prompty_template, self.chat_service)
            self.kernel.add_plugin(prompty_plugin, plugin_name="prompty")
            logger.info("✅ Added PromptyPlugin")
        
        except Exception as e:
            logger.error(f"Failed to add Prompty plugin: {e}")
    
    def is_configured(self) -> bool:
        """Check if client is properly configured"""
        return all([self.kernel, self.endpoint, self.api_key])
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get configuration status"""
        return {
            "semantic_kernel": {
                "available": SK_AVAILABLE,
                "initialized": self.kernel is not None,
                "plugins": list(self.kernel.plugins.keys()) if self.kernel else []
            },
            "prompty": {
                "available": PROMPTY_AVAILABLE,
                "template_loaded": hasattr(self, 'prompty_template')
            },
            "azure_openai": {
                "endpoint_set": bool(self.endpoint),
                "api_key_set": bool(self.api_key),
                "model": self.model_name
            },
            "health": self._get_health_status()
        }
    
    def _get_health_status(self) -> str:
        """Get overall health status"""
        if not SK_AVAILABLE:
            return "❌ Semantic Kernel not available"
        if not self.kernel:
            return "❌ Kernel not initialized"
        if not self.endpoint or not self.api_key:
            return "❌ Azure credentials missing"
        return "✅ Healthy"
    
    async def _preprocess_documents(self, documents: List[Any]) -> List[Any]:
        """Basic document preprocessing"""
        processed = []
        for doc in documents:
            if len(doc.content.strip()) < 50:
                logger.warning(f"Document {doc.filename} has short content: {len(doc.content)} chars")
            processed.append(doc)
        return processed
    
    async def generate_assessment_report(self, assessment_request: Any, use_prompty_plugin: bool = True) -> Dict[str, Any]:
        """Generate assessment report using Semantic Kernel"""
        start_time = datetime.now()
        
        if not self.is_configured():
            return {
                "success": False,
                "error": "Client not configured properly",
                "report": None
            }
        
        try:
            # Preprocess documents
            processed_documents = await self._preprocess_documents(assessment_request.documents)
            
            # Generate report
            if use_prompty_plugin and "prompty" in (self.kernel.plugins.keys() if self.kernel else []):
                report_content = await self._generate_with_prompty(processed_documents)
                method = "prompty"
            else:
                report_content = await self._generate_traditional(processed_documents)
                method = "traditional"
            
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "success": True,
                "report": str(report_content),
                "metadata": {
                    "session_id": assessment_request.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "method": method,
                    "model": self.model_name,
                    "documents_processed": len(processed_documents)
                },
                "performance": {
                    "total_time_ms": round(total_time, 2),
                    "status": "success"
                }
            }
            
        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Report generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "report": None,
                "performance": {
                    "total_time_ms": round(total_time, 2),
                    "status": "error"
                }
            }
    
    async def _generate_with_prompty(self, documents: List[Any]) -> str:
        """Generate report using Prompty plugin"""
        document_content = "\n\n---DOCUMENT---\n\n".join([
            f"DOCUMENT: {doc.filename}\nCONTENT:\n{doc.content}"
            for doc in documents
        ])
        
        prompty_plugin = self.kernel.plugins["prompty"]
        generate_function = prompty_plugin["generate_report"]
        result = await generate_function.invoke(self.kernel, documents=document_content)
        
        return str(result.value) if hasattr(result, 'value') else str(result)
    
    async def _generate_traditional(self, documents: List[Any]) -> str:
        """Generate report using traditional method"""
        if not self.prompt_generator:
            raise ValueError("Prompt generator not available")
        
        prompt = self.prompt_generator.create_assessment_prompt(documents)
        system_message = self.prompt_generator.create_system_message()
        
        chat_history = ChatHistory()
        chat_history.add_system_message(system_message)
        chat_history.add_user_message(prompt)
        
        result = await self.chat_service.get_chat_message_content(
            chat_history=chat_history,
            settings=PromptExecutionSettings(max_tokens=4000, temperature=0.3)
        )
        
        return str(result.content) if hasattr(result, 'content') else str(result)
    
    async def validate_documents(self, documents: List[Any]) -> Dict[str, Any]:
        """Validate documents using Semantic Kernel"""
        if not self.document_extractor:
            return {"valid": False, "errors": ["Document extractor not available"]}
        
        # Get base validation
        validation_result = await self.document_extractor.validate_documents(documents)
        
        # Enhance with Semantic Kernel if available
        if self.kernel and "assessment" in self.kernel.plugins:
            try:
                assessment_plugin = self.kernel.plugins["assessment"]
                document_content = "---DOCUMENT---".join([doc.content for doc in documents])
                analysis = await assessment_plugin["analyze_documents"].invoke(
                    self.kernel, documents=document_content
                )
                validation_result["sk_analysis"] = str(analysis.value) if hasattr(analysis, 'value') else str(analysis)
            except Exception as e:
                logger.warning(f"SK validation enhancement failed: {e}")
        
        return validation_result
    
    async def get_kernel_info(self) -> Dict[str, Any]:
        """Get kernel information"""
        if not self.kernel:
            return {"configured": False, "error": "Kernel not initialized"}
        
        return {
            "configured": True,
            "services": [service.service_id for service in self.kernel.services.values()],
            "plugins": list(self.kernel.plugins.keys()),
            "functions": {
                plugin_name: list(plugin.functions.keys()) 
                for plugin_name, plugin in self.kernel.plugins.items()
            }
        }


# Backward compatibility
try:
    from document_extractor import DocumentContent, AssessmentRequest, process_assessment_documents
except ImportError:
    class DocumentContent:
        pass
    
    class AssessmentRequest:
        pass
    
    def process_assessment_documents(*args, **kwargs):
        raise ImportError("Document processing not available")

# Alias for easier switching
AzureLLMClientSK = AzureLLMClientSemanticKernel