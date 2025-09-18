"""Generic Semantic Kernel Azure LLM Client"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

try:
    from semantic_kernel import Kernel
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
    from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
    from semantic_kernel.contents.chat_history import ChatHistory
    SK_AVAILABLE = True
except ImportError:
    logger.warning("Semantic Kernel not available. Install with: pip install semantic-kernel>=1.0.0")
    SK_AVAILABLE = False


class AzureLLMClientSemanticKernel:
    """Semantic Kernel-based Azure LLM client"""
    
    def __init__(self, document_processor=None, prompt_generator=None):
        # Load configuration
        self.endpoint = os.getenv("LLM_ENDPOINT")
        self.api_key = os.getenv("LLM_API_KEY")
        self.model_name = os.getenv("LLM_MODEL_NAME", "gpt-4")
        self.kernel = None
        self.chat_service = None
        
        if SK_AVAILABLE:
            self._initialize_kernel()
        
        # Load helper components
        self.document_extractor = document_processor or self._load_component("document_extractor", "DocumentExtractor")
        self.prompt_generator = prompt_generator or self._load_component("assessment_prompt", "AssessmentPromptGenerator")
    
    def _load_component(self, module_name, class_name):
        """Load optional component with error handling"""
        try:
            module = __import__(module_name)
            return getattr(module, class_name)()
        except ImportError:
            logger.info(f"{class_name} not available")
        return None
    
    def _initialize_kernel(self):
        """Initialize Semantic Kernel with Azure OpenAI"""
        if not (self.endpoint and self.api_key):
            logger.error("Azure OpenAI credentials not configured")
            return
        
        try:
            self.kernel = Kernel()
            self.chat_service = AzureChatCompletion(
                service_id="azure_openai_chat",
                deployment_name=self.model_name,
                endpoint=self.endpoint,
                api_key=self.api_key,
                api_version="2024-02-15-preview"
            )
            self.kernel.add_service(self.chat_service)
            logger.info("Semantic Kernel initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Semantic Kernel: {e}")
            self.kernel = None
    
    def is_configured(self) -> bool:
        """Check if client is properly configured"""
        return bool(self.kernel and self.endpoint and self.api_key)
    
    async def execute_template(self, template_variables: Dict[str, Any]) -> Dict[str, Any]:
        """Execute template with given variables"""
        if not self.is_configured():
            return {"success": False, "error": "Client not configured", "result": None}
        
        try:
            if self.prompt_generator and hasattr(self.prompt_generator, 'generate_from_template'):
                prompt = self.prompt_generator.generate_from_template(template_variables)
                system_message = getattr(self.prompt_generator, 'get_system_message', lambda: None)()
                response = await self.simple_completion(prompt, system_message)
            elif 'prompt' in template_variables:
                prompt = template_variables['prompt']
                system_message = template_variables.get('system_message')
                response = await self.simple_completion(prompt, system_message)
            else:
                # Fallback for assessment-specific workflow
                documents = template_variables.get('documents', '')
                
                class MockDoc:
                    def __init__(self, content):
                        self.filename = "template_documents.txt"
                        self.document_type = "Template Documents"
                        self.content = content
                        self.is_required = True
                
                mock_docs = [MockDoc(documents)] if documents else []
                prompt = self.prompt_generator.create_assessment_prompt(mock_docs)
                system_message = self.prompt_generator.create_system_message()
                full_prompt = f"{system_message}\n\n{prompt}"
                response = await self.chat_completion(prompt=full_prompt, max_tokens=4000, temperature=0.3)
            
            # Extract content and usage from response
            if isinstance(response, dict):
                result_content = response.get("content", response)
                token_usage = response.get("usage")
            else:
                result_content = response
                token_usage = None
            
            result = {
                "success": True,
                "result": result_content,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model_used": self.model_name,
                    "client_type": "semantic_kernel"
                }
            }
            
            # Add token usage if available
            if token_usage:
                # Handle both dictionary and object types for token usage
                if hasattr(token_usage, 'get'):
                    # Dictionary-like object
                    prompt_tokens = token_usage.get("prompt_tokens")
                    completion_tokens = token_usage.get("completion_tokens")
                    total_tokens = token_usage.get("total_tokens")
                    
                    # Calculate total_tokens if not provided
                    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                        total_tokens = prompt_tokens + completion_tokens
                    
                    result["metadata"]["total_tokens"] = total_tokens
                    result["metadata"]["prompt_tokens"] = prompt_tokens
                    result["metadata"]["completion_tokens"] = completion_tokens
                else:
                    # Object with attributes (CompletionUsage)
                    prompt_tokens = getattr(token_usage, "prompt_tokens", None)
                    completion_tokens = getattr(token_usage, "completion_tokens", None)
                    total_tokens = getattr(token_usage, "total_tokens", None)
                    
                    # Calculate total_tokens if not provided
                    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                        total_tokens = prompt_tokens + completion_tokens
                    
                    result["metadata"]["total_tokens"] = total_tokens
                    result["metadata"]["prompt_tokens"] = prompt_tokens
                    result["metadata"]["completion_tokens"] = completion_tokens
            
            return result
            
        except Exception as e:
            logger.error(f"Template execution failed: {e}")
            return {"success": False, "error": str(e), "result": None}
    
    async def chat_completion(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.3) -> Dict[str, Any]:
        """Direct chat completion method"""
        chat_history = ChatHistory()
        chat_history.add_user_message(prompt)
        
        settings = PromptExecutionSettings(max_tokens=max_tokens, temperature=temperature)
        result = await self.chat_service.get_chat_message_content(chat_history=chat_history, settings=settings)
        
        # Extract content
        content = str(result.content) if hasattr(result, 'content') else str(result)
        
        # Extract usage information if available
        usage = None
        if hasattr(result, 'metadata') and result.metadata:
            # Check for usage info in metadata
            usage_data = result.metadata.get('usage', None)
            if usage_data:
                # Handle both dictionary and object types for token usage
                if hasattr(usage_data, 'get'):
                    # Dictionary-like object
                    prompt_tokens = usage_data.get('prompt_tokens')
                    completion_tokens = usage_data.get('completion_tokens')
                    total_tokens = usage_data.get('total_tokens')
                    
                    # Calculate total_tokens if not provided
                    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                        total_tokens = prompt_tokens + completion_tokens
                    
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens, 
                        "total_tokens": total_tokens
                    }
                else:
                    # Object with attributes (CompletionUsage)
                    prompt_tokens = getattr(usage_data, 'prompt_tokens', None)
                    completion_tokens = getattr(usage_data, 'completion_tokens', None)
                    total_tokens = getattr(usage_data, 'total_tokens', None)
                    
                    # Calculate total_tokens if not provided
                    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                        total_tokens = prompt_tokens + completion_tokens
                    
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens, 
                        "total_tokens": total_tokens
                    }
        
        return {
            "content": content,
            "usage": usage
        }

    async def chat_completion_with_history(self, messages: List[Dict[str, str]], max_tokens: int = 2000, 
                                          temperature: float = 0.3) -> Dict[str, Any]:
        """Chat completion with message history"""
        chat_history = ChatHistory()
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                chat_history.add_system_message(content)
            elif role == 'user':
                chat_history.add_user_message(content)
            elif role == 'assistant':
                chat_history.add_assistant_message(content)
        
        settings = PromptExecutionSettings(max_tokens=max_tokens, temperature=temperature)
        result = await self.chat_service.get_chat_message_content(chat_history=chat_history, settings=settings)
        
        # Extract content
        content = str(result.content) if hasattr(result, 'content') else str(result)
        
        # Extract usage information if available
        usage = None
        if hasattr(result, 'metadata') and result.metadata:
            # Check for usage info in metadata
            usage_data = result.metadata.get('usage', None)
            if usage_data:
                # Handle both dictionary and object types for token usage
                if hasattr(usage_data, 'get'):
                    # Dictionary-like object
                    prompt_tokens = usage_data.get('prompt_tokens')
                    completion_tokens = usage_data.get('completion_tokens')
                    total_tokens = usage_data.get('total_tokens')
                    
                    # Calculate total_tokens if not provided
                    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                        total_tokens = prompt_tokens + completion_tokens
                    
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens, 
                        "total_tokens": total_tokens
                    }
                else:
                    # Object with attributes (CompletionUsage)
                    prompt_tokens = getattr(usage_data, 'prompt_tokens', None)
                    completion_tokens = getattr(usage_data, 'completion_tokens', None)
                    total_tokens = getattr(usage_data, 'total_tokens', None)
                    
                    # Calculate total_tokens if not provided
                    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                        total_tokens = prompt_tokens + completion_tokens
                    
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens, 
                        "total_tokens": total_tokens
                    }
        
        return {
            "content": content,
            "usage": usage
        }

    async def simple_completion(self, prompt: str, system_message: str = None, 
                               max_tokens: int = 2000, temperature: float = 0.3) -> Dict[str, Any]:
        """Simple text completion with optional system message"""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        return await self.chat_completion_with_history(messages, max_tokens, temperature)

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
            response = await self.simple_completion(prompt, system_message, max_tokens=4000, temperature=0.3)
            
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
                    "client_type": "semantic_kernel"
                },
                "error": None
            }
            
            # Add token usage if available
            if token_usage:
                # Handle both dictionary and object types for token usage
                if hasattr(token_usage, 'get'):
                    # Dictionary-like object
                    prompt_tokens = token_usage.get("prompt_tokens")
                    completion_tokens = token_usage.get("completion_tokens")
                    total_tokens = token_usage.get("total_tokens")
                    
                    # Calculate total_tokens if not provided
                    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                        total_tokens = prompt_tokens + completion_tokens
                    
                    result["metadata"]["total_tokens"] = total_tokens
                    result["metadata"]["prompt_tokens"] = prompt_tokens
                    result["metadata"]["completion_tokens"] = completion_tokens
                else:
                    # Object with attributes (CompletionUsage)
                    prompt_tokens = getattr(token_usage, "prompt_tokens", None)
                    completion_tokens = getattr(token_usage, "completion_tokens", None)
                    total_tokens = getattr(token_usage, "total_tokens", None)
                    
                    # Calculate total_tokens if not provided
                    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                        total_tokens = prompt_tokens + completion_tokens
                    
                    result["metadata"]["total_tokens"] = total_tokens
                    result["metadata"]["prompt_tokens"] = prompt_tokens
                    result["metadata"]["completion_tokens"] = completion_tokens
            
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
                    "client_type": "semantic_kernel"
                }
            }


# Alias for easier switching between implementations
AzureLLMClientSK = AzureLLMClientSemanticKernel