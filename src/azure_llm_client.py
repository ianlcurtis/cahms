"""
Simple Azure LLM Client for CAHMS Neurodevelopmental Assessment
This module handles interactions with Azure-hosted Large Language Models
for generating assessment reports.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# LLM API imports
try:
    from openai import AzureOpenAI
    import requests
except ImportError:
    print("Required packages not installed. Install with: pip install openai requests")
    AzureOpenAI = None
    requests = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureLLMClient:
    """Simplified client for interacting with Azure LLM services"""
    
    def __init__(self):
        """Initialize the LLM client with configuration from environment variables"""
        self.endpoint = os.getenv("LLM_ENDPOINT")
        self.api_key = os.getenv("LLM_API_KEY")
        self.model_name = os.getenv("LLM_MODEL_NAME", "gpt-4")
        self.use_openai_client = os.getenv("USE_OPENAI_CLIENT", "true").lower() == "true"
        
        # Initialize client
        self.client = None
        self._initialize_client()
        
        # Initialize helper components
        try:
            from document_extractor import DocumentExtractor
            from assessment_prompt import AssessmentPromptGenerator
            self.document_extractor = DocumentExtractor()
            self.prompt_generator = AssessmentPromptGenerator()
        except ImportError:
            logger.warning("Could not import helper modules. Some functionality may be limited.")
            self.document_extractor = None
            self.prompt_generator = None
    
    def _initialize_client(self):
        """Initialize the LLM client"""
        if not self.endpoint or not self.api_key:
            logger.warning("LLM credentials not configured. Set LLM_ENDPOINT and LLM_API_KEY")
            return
        
        try:
            if self.use_openai_client and AzureOpenAI:
                self.client = AzureOpenAI(
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                    api_version="2024-02-15-preview"
                )
                logger.info("Azure OpenAI client initialized successfully")
            else:
                logger.info("Using direct HTTP requests for LLM communication")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
    
    def is_configured(self) -> bool:
        """Check if the client is properly configured"""
        return (self.endpoint is not None and self.api_key is not None)
    
    def _make_direct_api_call(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Make a direct API call to the LLM endpoint"""
        if not requests:
            raise ImportError("requests library not available")
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
            "api-version": "2024-02-15-preview"
        }
        
        payload = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4000),
            "temperature": kwargs.get("temperature", 0.3),
            "top_p": kwargs.get("top_p", 0.9)
        }
        
        url = f"{self.endpoint.rstrip('/')}/openai/deployments/{self.model_name}/chat/completions"
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Direct API call failed: {e}")
            raise
    
    async def generate_assessment_report(self, assessment_request: Any) -> Dict[str, Any]:
        """Generate a neurodevelopmental assessment report using LLM"""
        if not self.is_configured():
            return {
                "success": False,
                "error": "LLM client not configured properly",
                "report": None
            }
        
        if not self.prompt_generator:
            return {
                "success": False,
                "error": "Prompt generator not available",
                "report": None
            }
        
        try:
            # Create the assessment prompt
            prompt = self.prompt_generator.create_assessment_prompt(assessment_request.documents)
            system_message = self.prompt_generator.create_system_message()
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            # Call LLM
            if self.client and self.use_openai_client:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=4000,
                    temperature=0.3,
                    top_p=0.9
                )
                report_content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            else:
                response = self._make_direct_api_call(
                    messages=messages,
                    max_tokens=4000,
                    temperature=0.3,
                    top_p=0.9
                )
                report_content = response['choices'][0]['message']['content']
                tokens_used = response.get('usage', {}).get('total_tokens')
            
            # Create response with metadata
            result = {
                "success": True,
                "report": report_content,
                "metadata": {
                    "session_id": assessment_request.session_id,
                    "generation_timestamp": datetime.now().isoformat(),
                    "model_used": self.model_name,
                    "documents_processed": len(assessment_request.documents),
                    "required_documents": len([d for d in assessment_request.documents if d.is_required]),
                    "optional_documents": len([d for d in assessment_request.documents if not d.is_required]),
                    "tokens_used": tokens_used
                },
                "error": None
            }
            
            logger.info(f"Successfully generated assessment report for session {assessment_request.session_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating assessment report: {e}")
            return {
                "success": False,
                "error": str(e),
                "report": None,
                "metadata": {
                    "session_id": assessment_request.session_id,
                    "generation_timestamp": datetime.now().isoformat(),
                    "error_timestamp": datetime.now().isoformat()
                }
            }
    
    async def validate_documents(self, documents: List[Any]) -> Dict[str, Any]:
        """Validate that uploaded documents contain appropriate content"""
        if not self.document_extractor:
            return {
                "valid": False,
                "errors": ["Document extractor not available"],
                "warnings": [],
                "document_analysis": {}
            }
        
        return await self.document_extractor.validate_documents(documents)


# Re-export the necessary classes and functions for backward compatibility
try:
    from document_extractor import DocumentContent, AssessmentRequest, process_assessment_documents
except ImportError:
    # Import from the old file for backward compatibility
    from azure_llm_client_old import DocumentContent, AssessmentRequest, process_assessment_documents
