"""
Semantic Kernel Azure LLM Client for CAHMS Neurodevelopmental Assessment
This module demonstrates using Microsoft Semantic Kernel for interactions with 
Azure-hosted Large Language Models for generating assessment reports.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Semantic Kernel imports
try:
    import semantic_kernel as sk
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
    from semantic_kernel.core_plugins.text_plugin import TextPlugin
    from semantic_kernel.functions.kernel_function_decorator import kernel_function
    from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
    from semantic_kernel.contents.chat_history import ChatHistory
    SK_AVAILABLE = True
    print("✅ Semantic Kernel imported successfully")
except ImportError as e:
    print(f"❌ Semantic Kernel not available: {e}")
    print("Install with: pip install semantic-kernel")
    sk = None
    AzureChatCompletion = None
    TextPlugin = None
    kernel_function = None
    PromptExecutionSettings = None
    ChatHistory = None
    SK_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureLLMClientSemanticKernel:
    """Semantic Kernel-based client for interacting with Azure LLM services"""
    
    def __init__(self):
        """Initialize the LLM client with Semantic Kernel"""
        self.endpoint = os.getenv("LLM_ENDPOINT")
        self.api_key = os.getenv("LLM_API_KEY")
        self.model_name = os.getenv("LLM_MODEL_NAME", "gpt-4")
        
        # Initialize Semantic Kernel
        self.kernel = None
        self.chat_service = None
        self._initialize_kernel()
        
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
    
    def _initialize_kernel(self):
        """Initialize the Semantic Kernel with Azure OpenAI service"""
        if not self.endpoint or not self.api_key:
            logger.warning("LLM credentials not configured. Set LLM_ENDPOINT and LLM_API_KEY")
            return
        
        if not SK_AVAILABLE:
            logger.error("Semantic Kernel not available")
            return
        
        try:
            # Create kernel
            self.kernel = sk.Kernel()
            
            # Add Azure OpenAI chat completion service
            self.chat_service = AzureChatCompletion(
                service_id="azure_openai_chat",
                deployment_name=self.model_name,
                endpoint=self.endpoint,
                api_key=self.api_key,
                api_version="2024-02-15-preview"
            )
            
            # Add the service to kernel
            self.kernel.add_service(self.chat_service)
            
            # Add built-in plugins
            self.kernel.add_plugin(TextPlugin(), plugin_name="text")
            
            # Create and register custom assessment plugin
            self._register_assessment_plugin()
            
            logger.info("Semantic Kernel initialized successfully with Azure OpenAI")
            
        except Exception as e:
            logger.error(f"Failed to initialize Semantic Kernel: {e}")
            self.kernel = None
    
    def _register_assessment_plugin(self):
        """Register custom assessment functions as a Semantic Kernel plugin"""
        if not self.kernel:
            return
        
        # Create assessment plugin class
        class AssessmentPlugin:
            """Custom plugin for neurodevelopmental assessment functions"""
            
            @kernel_function(
                description="Analyze neurodevelopmental assessment documents and generate insights",
                name="analyze_documents"
            )
            def analyze_documents(self, documents: str) -> str:
                """Analyze assessment documents for key insights"""
                # This could include document validation, key information extraction, etc.
                return f"Analyzed {len(documents.split('---DOCUMENT---'))} documents for assessment insights"
            
            @kernel_function(
                description="Generate structured assessment report sections",
                name="generate_report_section"
            )
            def generate_report_section(self, section_type: str, content: str) -> str:
                """Generate a specific section of the assessment report"""
                return f"Generated {section_type} section based on provided content"
        
        # Register the plugin
        assessment_plugin = AssessmentPlugin()
        self.kernel.add_plugin(assessment_plugin, plugin_name="assessment")
    
    def is_configured(self) -> bool:
        """Check if the client is properly configured"""
        return (self.kernel is not None and 
                self.endpoint is not None and 
                self.api_key is not None)
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get detailed configuration status for debugging"""
        return {
            "sk_available": SK_AVAILABLE,
            "endpoint_set": bool(self.endpoint),
            "api_key_set": bool(self.api_key),
            "model_name": self.model_name,
            "kernel_initialized": self.kernel is not None,
            "environment_variables": {
                "LLM_ENDPOINT": "✅ Set" if os.getenv("LLM_ENDPOINT") else "❌ Not set",
                "LLM_API_KEY": "✅ Set" if os.getenv("LLM_API_KEY") else "❌ Not set", 
                "LLM_MODEL_NAME": f"✅ Set to '{os.getenv('LLM_MODEL_NAME', 'gpt-4')}'" if os.getenv("LLM_MODEL_NAME") else "⚠️  Using default 'gpt-4'",
                "USE_OPENAI_CLIENT": f"✅ Set to '{os.getenv('USE_OPENAI_CLIENT', 'true')}'" if os.getenv("USE_OPENAI_CLIENT") else "⚠️  Using default 'true'"
            }
        }
    
    async def _preprocess_documents(self, documents: List[Any]) -> List[Any]:
        """Preprocess documents to check for issues before sending to LLM"""
        print(f"\n🔍 DEBUG (SK): Preprocessing {len(documents)} documents")
        processed_documents = []
        
        for i, doc in enumerate(documents):
            print(f"🔍 DEBUG (SK): Processing document {i+1}: {doc.filename}")
            
            # Check if document has processing errors
            if doc.content.startswith('[') and doc.content.endswith(']'):
                logger.warning(f"Document {doc.filename} has processing error: {doc.content}")
                print(f"⚠ DEBUG (SK): Document {i+1} has processing error: {doc.content}")
                processed_documents.append(doc)
            else:
                # Check for suspiciously short content
                if len(doc.content.strip()) < 50:
                    logger.warning(f"Document {doc.filename} has very short content: {len(doc.content)} characters")
                    print(f"⚠ DEBUG (SK): Document {i+1} has very short content: {len(doc.content)} characters")
                
                # Use Semantic Kernel text plugin for additional analysis if available
                if self.kernel:
                    try:
                        # Example: Use text plugin to get word count
                        text_plugin = self.kernel.get_plugin("text")
                        if text_plugin and hasattr(text_plugin, "length"):
                            char_count_result = await text_plugin["length"].invoke(
                                self.kernel, 
                                text=doc.content
                            )
                            char_count = str(char_count_result.value) if hasattr(char_count_result, 'value') else str(char_count_result)
                            print(f"✅ DEBUG (SK): Document {i+1} analyzed with SK ({char_count} chars)")
                    except Exception as e:
                        print(f"⚠ DEBUG (SK): Could not analyze document with SK: {e}")
                
                print(f"✅ DEBUG (SK): Document {i+1} looks good ({len(doc.content)} chars)")
                processed_documents.append(doc)
        
        print(f"🔍 DEBUG (SK): Preprocessing complete, {len(processed_documents)} documents ready")
        return processed_documents
    
    async def generate_assessment_report(self, assessment_request: Any) -> Dict[str, Any]:
        """Generate a neurodevelopmental assessment report using Semantic Kernel"""
        if not self.is_configured():
            return {
                "success": False,
                "error": "Semantic Kernel LLM client not configured properly",
                "report": None
            }
        
        if not self.prompt_generator:
            return {
                "success": False,
                "error": "Prompt generator not available",
                "report": None
            }
        
        try:
            # Preprocess documents to check for issues
            processed_documents = await self._preprocess_documents(assessment_request.documents)
            
            # Create the assessment prompt
            prompt = self.prompt_generator.create_assessment_prompt(processed_documents)
            system_message = self.prompt_generator.create_system_message()
            
            # Create a prompt for assessment report generation
            full_prompt = f"""
            {system_message}
            
            {prompt}
            """
            
            # Configure prompt execution settings
            execution_settings = PromptExecutionSettings(
                service_id="azure_openai_chat",
                max_tokens=4000,
                temperature=0.3,
                top_p=0.9
            )
            
            # Create chat history and add the prompt
            chat_history = ChatHistory()
            chat_history.add_user_message(full_prompt)
            
            # Invoke the chat completion service directly
            result = await self.chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=execution_settings
            )
            
            report_content = str(result.content) if hasattr(result, 'content') else str(result)
            
            # Try to get token usage from the result (if available)
            tokens_used = None
            if hasattr(result, 'metadata') and result.metadata:
                usage_info = result.metadata.get('usage', {})
                if usage_info:
                    # Handle both dictionary and CompletionUsage object
                    if hasattr(usage_info, 'get'):
                        # It's a dictionary
                        tokens_used = usage_info.get('total_tokens')
                    elif hasattr(usage_info, 'prompt_tokens') and hasattr(usage_info, 'completion_tokens'):
                        # It's a CompletionUsage object - calculate total tokens
                        tokens_used = usage_info.prompt_tokens + usage_info.completion_tokens
                    else:
                        tokens_used = None
            
            # Create response with metadata
            response = {
                "success": True,
                "report": report_content,
                "metadata": {
                    "session_id": assessment_request.session_id,
                    "generation_timestamp": datetime.now().isoformat(),
                    "model_used": self.model_name,
                    "documents_processed": len(assessment_request.documents),
                    "required_documents": len([d for d in assessment_request.documents if d.is_required]),
                    "optional_documents": len([d for d in assessment_request.documents if not d.is_required]),
                    "tokens_used": tokens_used,
                    "using_semantic_kernel": True,
                    "kernel_plugins": list(self.kernel.plugins.keys()) if self.kernel else []
                },
                "error": None
            }
            
            logger.info(f"Successfully generated assessment report using Semantic Kernel for session {assessment_request.session_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating assessment report with Semantic Kernel: {e}")
            return {
                "success": False,
                "error": str(e),
                "report": None,
                "metadata": {
                    "session_id": assessment_request.session_id,
                    "generation_timestamp": datetime.now().isoformat(),
                    "error_timestamp": datetime.now().isoformat(),
                    "using_semantic_kernel": True
                }
            }
    
    async def validate_documents(self, documents: List[Any]) -> Dict[str, Any]:
        """Validate that uploaded documents contain appropriate content using Semantic Kernel"""
        if not self.document_extractor:
            return {
                "valid": False,
                "errors": ["Document extractor not available"],
                "warnings": [],
                "document_analysis": {}
            }
        
        # Get base validation
        validation_result = await self.document_extractor.validate_documents(documents)
        
        # Add Semantic Kernel enhancements if available
        if self.kernel:
            try:
                assessment_plugin = self.kernel.get_plugin("assessment")
                if assessment_plugin:
                    # Use custom assessment plugin for additional validation
                    document_content = "---DOCUMENT---".join([doc.content for doc in documents])
                    analysis_result = await assessment_plugin["analyze_documents"].invoke(
                        self.kernel,
                        documents=document_content
                    )
                    analysis = str(analysis_result.value) if hasattr(analysis_result, 'value') else str(analysis_result)
                    validation_result["sk_analysis"] = analysis
                    validation_result["enhanced_with_sk"] = True
            except Exception as e:
                logger.warning(f"Could not enhance validation with Semantic Kernel: {e}")
                validation_result["sk_analysis"] = f"Analysis failed: {e}"
                validation_result["enhanced_with_sk"] = False
        
        return validation_result
    
    async def get_kernel_info(self) -> Dict[str, Any]:
        """Get information about the current Semantic Kernel configuration"""
        if not self.kernel:
            return {"configured": False, "error": "Kernel not initialized"}
        
        try:
            return {
                "configured": True,
                "services": [service.service_id for service in self.kernel.services.values()],
                "plugins": list(self.kernel.plugins.keys()),
                "functions": {
                    plugin_name: list(plugin.functions.keys()) 
                    for plugin_name, plugin in self.kernel.plugins.items()
                },
                "model_name": self.model_name,
                "endpoint_configured": bool(self.endpoint)
            }
        except Exception as e:
            return {"configured": False, "error": str(e)}


# Re-export the necessary classes and functions for backward compatibility
try:
    from document_extractor import DocumentContent, AssessmentRequest, process_assessment_documents
except ImportError:
    # Define minimal classes if import fails
    class DocumentContent:
        pass
    
    class AssessmentRequest:
        pass
    
    def process_assessment_documents(*args, **kwargs):
        raise ImportError("Document processing not available")

# Alias for easier switching between implementations
AzureLLMClientSK = AzureLLMClientSemanticKernel