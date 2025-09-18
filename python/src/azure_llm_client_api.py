"""Generic Azure LLM Client using Azure OpenAI API"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

try:
    from openai import AzureOpenAI
    import requests
    API_AVAILABLE = True
except ImportError:
    logger.warning("Install required packages: pip install openai requests")
    AzureOpenAI = requests = None
    API_AVAILABLE = False


class AzureLLMClient:
    """Azure LLM client with OpenAI API and direct HTTP support"""
    
    def __init__(self, document_processor=None, prompt_generator=None):
        # Load configuration
        self.endpoint = os.getenv("LLM_ENDPOINT")
        self.api_key = os.getenv("LLM_API_KEY")
        self.model_name = os.getenv("LLM_MODEL_NAME", "gpt-4")
        self.use_openai_client = os.getenv("USE_OPENAI_CLIENT", "true").lower() == "true"
        
        # Initialize client
        self.client = self._init_client()
        
        # Load helper components
        self.document_extractor = document_processor or self._load_component("document_extractor", "DocumentExtractor")
        self.prompt_generator = prompt_generator or self._load_component("assessment_prompt", "AssessmentPromptGenerator")
    
    def _init_client(self):
        """Initialize Azure OpenAI client if configured"""
        if not (self.endpoint and self.api_key and API_AVAILABLE):
            logger.warning("LLM client not fully configured")
            return None
        
        try:
            if self.use_openai_client:
                return AzureOpenAI(
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                    api_version="2024-02-15-preview"
                )
            logger.info("Using direct HTTP requests")
        except Exception as e:
            logger.error(f"Client initialization failed: {e}")
        return None
    
    def _load_component(self, module_name, class_name):
        """Load optional component with error handling"""
        try:
            module = __import__(module_name)
            return getattr(module, class_name)()
        except ImportError:
            logger.info(f"{class_name} not available")
        return None
    
    def is_configured(self) -> bool:
        """Check if client is properly configured"""
        return bool(self.endpoint and self.api_key)
    
    def _make_direct_api_call(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Make direct API call to Azure OpenAI endpoint"""
        if not requests:
            raise ImportError("requests library required for direct API calls")
        
        url = f"{self.endpoint.rstrip('/')}/openai/deployments/{self.model_name}/chat/completions"
        payload = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4000),
            "temperature": kwargs.get("temperature", 0.3),
            "top_p": kwargs.get("top_p", 0.9)
        }
        
        response = requests.post(
            url,
            headers={"Content-Type": "application/json", "api-key": self.api_key},
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    
    async def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 4000, 
                             temperature: float = 0.3, **kwargs) -> Dict[str, Any]:
        """Generate chat completion using Azure OpenAI"""
        if not self.is_configured():
            raise ValueError("LLM client not configured")
        
        try:
            if self.client and self.use_openai_client:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                return {
                    "content": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    } if response.usage else None
                }
            else:
                response = self._make_direct_api_call(
                    messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs
                )
                return {
                    "content": response['choices'][0]['message']['content'],
                    "usage": response.get('usage', None)
                }
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    async def simple_completion(self, prompt: str, system_message: str = None,
                               max_tokens: int = 4000, temperature: float = 0.3, **kwargs) -> Dict[str, Any]:
        """Simple completion wrapper"""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        return await self.chat_completion(messages, max_tokens, temperature, **kwargs)
    
    async def execute_template(self, template_variables: Dict[str, Any]) -> Dict[str, Any]:
        """Execute template with variables (requires prompt generator)"""
        if not self.is_configured():
            return {"success": False, "error": "Client not configured", "result": None}
        
        try:
            if self.prompt_generator and hasattr(self.prompt_generator, 'generate_from_template'):
                prompt = self.prompt_generator.generate_from_template(template_variables)
                system_message = getattr(self.prompt_generator, 'get_system_message', lambda: None)()
            elif 'prompt' in template_variables:
                prompt = template_variables['prompt']
                system_message = template_variables.get('system_message')
            else:
                return {"success": False, "error": "No prompt generator or 'prompt' key", "result": None}
            
            result = await self.simple_completion(prompt, system_message)
            return {
                "success": True,
                "result": result.get("content", result) if isinstance(result, dict) else result,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model_used": self.model_name,
                    "client_type": "azure_api",
                    "token_usage": result.get("usage") if isinstance(result, dict) else None
                }
            }
        except Exception as e:
            logger.error(f"Template execution failed: {e}")
            return {"success": False, "error": str(e), "result": None}
    
    async def generate_response(self, request_prompt: Any) -> Dict[str, Any]:
        """Generate response for backward compatibility with assessment workflows"""
        if not self.is_configured():
            return {"success": False, "error": "Client not configured", "report": None}
        
        try:
            # Check if prompt is already prepared (new workflow)
            if hasattr(request_prompt, 'prompt'):
                prompt = request_prompt.prompt
                system_message = getattr(request_prompt, 'system_message', None)
            # Fallback to old workflow with prompt generator
            elif self.prompt_generator and hasattr(request_prompt, 'documents'):
                # Process documents if available
                processed_docs = None
                if self.document_extractor:
                    processed_docs = await self.document_extractor.preprocess_documents(request_prompt.documents)
                else:
                    processed_docs = request_prompt.documents
                
                # Generate prompt using the old method
                if hasattr(self.prompt_generator, 'create_assessment_prompt') and processed_docs:
                    prompt = self.prompt_generator.create_assessment_prompt(processed_docs)
                    system_message = self.prompt_generator.create_system_message()
                else:
                    return {"success": False, "error": "Cannot generate prompt with prompt generator", "report": None}
            else:
                return {"success": False, "error": "No prompt provided and prompt generator not available", "report": None}
            
            # Generate response
            response = await self.simple_completion(prompt, system_message)
            
            # Extract content and usage from response
            if isinstance(response, dict):
                report_content = response.get("content", "")
                token_usage = response.get("usage")
            else:
                report_content = response
                token_usage = None
            
            # Build result with metadata
            result = {
                "success": True,
                "report": report_content,
                "metadata": {
                    "session_id": getattr(request_prompt, 'session_id', 'unknown'),
                    "generation_timestamp": datetime.now().isoformat(),
                    "model_used": self.model_name,
                    "client_type": "azure_api"
                },
                "error": None
            }
            
            # Add token usage if available
            if token_usage:
                # Handle both dictionary and object types for token usage
                if hasattr(token_usage, 'get'):
                    # Dictionary-like object
                    result["metadata"]["total_tokens"] = token_usage.get("total_tokens")
                    result["metadata"]["prompt_tokens"] = token_usage.get("prompt_tokens")
                    result["metadata"]["completion_tokens"] = token_usage.get("completion_tokens")
                else:
                    # Object with attributes (CompletionUsage)
                    result["metadata"]["total_tokens"] = getattr(token_usage, "total_tokens", None)
                    result["metadata"]["prompt_tokens"] = getattr(token_usage, "prompt_tokens", None)
                    result["metadata"]["completion_tokens"] = getattr(token_usage, "completion_tokens", None)
            
            # Add document metadata if available
            documents = getattr(request_prompt, 'documents', None)
            if documents:
                result["metadata"].update({
                    "documents_processed": len(documents),
                    "required_documents": len([d for d in documents if getattr(d, 'is_required', False)]),
                    "optional_documents": len([d for d in documents if not getattr(d, 'is_required', True)])
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "report": None,
                "metadata": {
                    "session_id": getattr(request_prompt, 'session_id', 'unknown'),
                    "generation_timestamp": datetime.now().isoformat(),
                    "client_type": "azure_api"
                }
            }
