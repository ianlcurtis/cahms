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

# Semantic Kernel imports with comprehensive error handling
try:
    import semantic_kernel as sk
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
    from semantic_kernel.core_plugins.text_plugin import TextPlugin
    from semantic_kernel.functions.kernel_function_decorator import kernel_function
    from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
    from semantic_kernel.contents.chat_history import ChatHistory
    from semantic_kernel import Kernel
    SK_AVAILABLE = True
    SK_VERSION = getattr(sk, '__version__', 'unknown')
    print(f"✅ Semantic Kernel v{SK_VERSION} imported successfully")
except ImportError as e:
    print(f"❌ Semantic Kernel not available: {e}")
    print("Install with: pip install semantic-kernel>=1.0.0")
    sk = None
    AzureChatCompletion = None
    TextPlugin = None
    kernel_function = None
    PromptExecutionSettings = None
    ChatHistory = None
    Kernel = None
    SK_AVAILABLE = False
    SK_VERSION = None

# Prompty imports with error handling
try:
    import prompty
    from prompty import load, prepare, execute
    PROMPTY_AVAILABLE = True
    PROMPTY_VERSION = getattr(prompty, '__version__', 'unknown')
    print(f"✅ Prompty v{PROMPTY_VERSION} imported successfully")
except ImportError as e:
    print(f"❌ Prompty not available: {e}")
    print("Install with: pip install prompty>=1.0.0")
    prompty = None
    load = None
    prepare = None
    execute = None
    PROMPTY_AVAILABLE = False
    PROMPTY_VERSION = None

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
        """Initialize the Semantic Kernel with Azure OpenAI service using best practices"""
        if not self.endpoint or not self.api_key:
            logger.warning("LLM credentials not configured. Set LLM_ENDPOINT and LLM_API_KEY")
            return
        
        if not SK_AVAILABLE:
            logger.error("Semantic Kernel not available")
            return
        
        try:
            # Create kernel with builder pattern (SK 1.0+ best practice)
            from semantic_kernel import Kernel
            self.kernel = Kernel()
            
            # Add Azure OpenAI chat completion service with proper configuration
            self.chat_service = AzureChatCompletion(
                service_id="azure_openai_chat",
                deployment_name=self.model_name,
                endpoint=self.endpoint,
                api_key=self.api_key,
                api_version="2024-02-15-preview"
            )
            
            # Add the service to kernel
            self.kernel.add_service(self.chat_service)
            
            # Add built-in plugins with error handling
            try:
                from semantic_kernel.core_plugins.text_plugin import TextPlugin
                self.kernel.add_plugin(TextPlugin(), plugin_name="text")
                logger.info("✅ Added TextPlugin to kernel")
            except Exception as e:
                logger.warning(f"Could not add TextPlugin: {e}")
            
            # Add built-in conversation summary plugin
            try:
                from semantic_kernel.core_plugins.conversation_summary_plugin import ConversationSummaryPlugin
                self.kernel.add_plugin(ConversationSummaryPlugin(), plugin_name="conversation_summary")
                logger.info("✅ Added ConversationSummaryPlugin to kernel")
            except Exception as e:
                logger.warning(f"Could not add ConversationSummaryPlugin: {e}")
            
            # Create and register custom assessment plugin
            self._register_assessment_plugin()
            
            # Create and register Prompty-based functions
            self._create_prompty_functions()
            
            logger.info("✅ Semantic Kernel initialized successfully with Azure OpenAI")
            
        except Exception as e:
            logger.error(f"Failed to initialize Semantic Kernel: {e}")
            self.kernel = None
    
    def _register_assessment_plugin(self):
        """Register custom assessment functions as a Semantic Kernel plugin using best practices"""
        if not self.kernel:
            return
        
        # Create assessment plugin class with improved functionality
        class AssessmentPlugin:
            """Custom plugin for neurodevelopmental assessment functions"""
            
            def __init__(self, kernel):
                self.kernel = kernel
            
            @kernel_function(
                description="Analyze neurodevelopmental assessment documents and extract key insights",
                name="analyze_documents"
            )
            async def analyze_documents(self, documents: str) -> str:
                """Analyze assessment documents for key insights using LLM"""
                try:
                    # Create a focused analysis prompt
                    analysis_prompt = f"""
                    Analyze the following neurodevelopmental assessment documents and extract key insights:
                    
                    Documents:
                    {documents}
                    
                    Provide:
                    1. Number of documents analyzed
                    2. Types of assessments found (Form S, Form H, Form A, etc.)
                    3. Key developmental concerns identified
                    4. Strengths noted
                    5. Recommendations for further assessment
                    
                    Be concise and clinical in your analysis.
                    """
                    
                    # Use the kernel's chat service for analysis
                    chat_history = ChatHistory()
                    chat_history.add_user_message(analysis_prompt)
                    
                    # Get chat service from kernel
                    chat_service = None
                    for service in self.kernel.services.values():
                        if hasattr(service, 'get_chat_message_content'):
                            chat_service = service
                            break
                    
                    if chat_service:
                        result = await chat_service.get_chat_message_content(
                            chat_history=chat_history,
                            settings=PromptExecutionSettings(
                                max_tokens=1000,
                                temperature=0.2
                            )
                        )
                        return str(result.content) if hasattr(result, 'content') else str(result)
                    else:
                        return f"Analyzed {len(documents.split('---DOCUMENT---'))} documents for assessment insights"
                        
                except Exception as e:
                    logger.error(f"Document analysis failed: {e}")
                    return f"Analysis failed: {e}"
            
            @kernel_function(
                description="Generate structured assessment report sections with clinical focus",
                name="generate_report_section"
            )
            async def generate_report_section(self, section_type: str, content: str) -> str:
                """Generate a specific section of the assessment report"""
                try:
                    section_prompt = f"""
                    Generate a {section_type} section for a neurodevelopmental assessment report based on:
                    
                    Content: {content}
                    
                    Ensure the section is:
                    - Clinically appropriate
                    - Evidence-based
                    - Professional in tone
                    - Structured with clear headings if needed
                    """
                    
                    # Use kernel's chat service
                    chat_history = ChatHistory()
                    chat_history.add_user_message(section_prompt)
                    
                    chat_service = None
                    for service in self.kernel.services.values():
                        if hasattr(service, 'get_chat_message_content'):
                            chat_service = service
                            break
                    
                    if chat_service:
                        result = await chat_service.get_chat_message_content(
                            chat_history=chat_history,
                            settings=PromptExecutionSettings(
                                max_tokens=800,
                                temperature=0.3
                            )
                        )
                        return str(result.content) if hasattr(result, 'content') else str(result)
                    else:
                        return f"Generated {section_type} section based on provided content"
                        
                except Exception as e:
                    logger.error(f"Section generation failed: {e}")
                    return f"Section generation failed: {e}"
            
            @kernel_function(
                description="Validate document completeness and quality for assessment",
                name="validate_document_quality"
            )
            def validate_document_quality(self, documents: str) -> str:
                """Validate document quality and completeness"""
                try:
                    doc_sections = documents.split('---DOCUMENT---')
                    validation_results = []
                    
                    required_forms = ['Form S', 'Form H', 'Form A']
                    found_forms = []
                    
                    for doc in doc_sections:
                        if any(form in doc for form in required_forms):
                            found_forms.extend([form for form in required_forms if form in doc])
                        
                        # Check document length
                        if len(doc.strip()) < 50:
                            validation_results.append("⚠️ Very short document detected")
                        elif len(doc.strip()) > 5000:
                            validation_results.append("✅ Comprehensive document")
                        else:
                            validation_results.append("✅ Adequate document length")
                    
                    # Summary
                    summary = f"📊 Analysis: {len(doc_sections)} documents, {len(set(found_forms))} standard forms found"
                    if found_forms:
                        summary += f" ({', '.join(set(found_forms))})"
                    
                    return f"{summary}\n" + "\n".join(validation_results)
                    
                except Exception as e:
                    return f"Validation failed: {e}"
        
        # Register the plugin with the kernel
        assessment_plugin = AssessmentPlugin(self.kernel)
        self.kernel.add_plugin(assessment_plugin, plugin_name="assessment")
        logger.info("✅ Registered enhanced AssessmentPlugin with kernel")
    
    def _create_prompty_functions(self):
        """Create Semantic Kernel functions from Prompty templates"""
        if not self.kernel:
            return
        
        try:
            # Import prompty functions with proper error handling
            from prompty import load, prepare, execute
            from prompty.azure import AzureOpenAIExecutor
            
            # Load the prompty template
            prompty_file_path = os.path.join(os.path.dirname(__file__), "assessment_report.prompty")
            
            if not os.path.exists(prompty_file_path):
                logger.warning(f"Prompty file not found: {prompty_file_path}")
                return
            
            # Load the template using Prompty SDK
            self.prompty_template = load(prompty_file_path)
            logger.info(f"Loaded Prompty template from {prompty_file_path}")
            
            # Configure Azure executor for Prompty
            self.prompty_executor = AzureOpenAIExecutor(
                endpoint=self.endpoint,
                api_key=self.api_key,
                api_version="2024-02-15-preview"
            )
            
            # Create Prompty-based plugin class with best practices
            class PromptyPlugin:
                """Plugin for Prompty-based assessment functions using SDK"""
                
                def __init__(self, template, chat_service, execution_settings, endpoint=None, api_key=None):
                    self.template = template
                    self.chat_service = chat_service
                    self.execution_settings = execution_settings
                    # Store credentials for Prompty executor
                    if endpoint and api_key:
                        try:
                            from prompty.azure import AzureOpenAIExecutor
                            self.prompty_executor = AzureOpenAIExecutor(
                                endpoint=endpoint,
                                api_key=api_key,
                                api_version="2024-02-15-preview"
                            )
                        except ImportError:
                            logger.warning("AzureOpenAIExecutor not available, using fallback method")
                            self.prompty_executor = None
                    else:
                        self.prompty_executor = None
                
                @kernel_function(
                    description="Generate CAHMS assessment report using Prompty template",
                    name="generate_assessment_report"
                )
                async def generate_assessment_report(self, documents: str) -> str:
                    """Generate assessment report using Prompty SDK through Semantic Kernel"""
                    try:
                        # First try using Prompty SDK with proper execution
                        if self.prompty_executor and self.template:
                            from prompty import prepare, execute
                            prepared_template = prepare(self.template, documents=documents)
                            result = await execute(prepared_template, executor=self.prompty_executor)
                            return str(result)
                        else:
                            # Use fallback method
                            return await self._fallback_execution(documents)
                        
                    except Exception as e:
                        logger.error(f"Prompty SDK execution failed: {e}")
                        # Fallback to manual parsing for backward compatibility
                        return await self._fallback_execution(documents)
                
                async def _fallback_execution(self, documents: str) -> str:
                    """Fallback method using manual template parsing"""
                    try:
                        # Read the template file directly
                        prompty_file_path = os.path.join(os.path.dirname(__file__), "assessment_report.prompty")
                        with open(prompty_file_path, 'r') as f:
                            template_content = f.read()
                        
                        # Extract system and user sections using improved parsing
                        system_message, user_template = self._parse_prompty_template(template_content)
                        user_prompt = user_template.replace('{{documents}}', documents)
                        
                        # Execute through Semantic Kernel's chat service
                        chat_history = ChatHistory()
                        chat_history.add_system_message(system_message)
                        chat_history.add_user_message(user_prompt)
                        
                        result = await self.chat_service.get_chat_message_content(
                            chat_history=chat_history,
                            settings=self.execution_settings
                        )
                        
                        return str(result.content) if hasattr(result, 'content') else str(result)
                        
                    except Exception as e:
                        logger.error(f"Fallback execution failed: {e}")
                        return f"Error generating report: {e}"
                
                def _parse_prompty_template(self, template_content: str) -> tuple[str, str]:
                    """Improved parsing of Prompty template format"""
                    lines = template_content.split('\n')
                    sections = {'system': [], 'user': []}
                    current_section = None
                    in_frontmatter = False
                    
                    for line in lines:
                        # Skip YAML frontmatter
                        if line.strip() == '---':
                            in_frontmatter = not in_frontmatter
                            continue
                        if in_frontmatter:
                            continue
                            
                        # Detect section headers
                        if line.strip() == 'system:':
                            current_section = 'system'
                            continue
                        elif line.strip() == 'user:':
                            current_section = 'user'
                            continue
                        
                        # Add content to current section
                        if current_section and current_section in sections:
                            sections[current_section].append(line)
                    
                    system_message = '\n'.join(sections['system']).strip()
                    user_template = '\n'.join(sections['user']).strip()
                    
                    return system_message, user_template
                
                @kernel_function(
                    description="Extract system message from Prompty template",
                    name="get_system_message"
                )
                def get_system_message(self) -> str:
                    """Extract the system message from the Prompty template"""
                    try:
                        # Read the template file directly
                        prompty_file_path = os.path.join(os.path.dirname(__file__), "assessment_report.prompty")
                        with open(prompty_file_path, 'r') as f:
                            template_content = f.read()
                        
                        # Extract system message from template content
                        lines = template_content.split('\n')
                        system_lines = []
                        in_system = False
                        
                        for line in lines:
                            if line.strip() == 'system:':
                                in_system = True
                                continue
                            elif line.strip() == 'user:' or (line.strip().startswith('---') and in_system):
                                break
                            elif in_system:
                                system_lines.append(line)
                                
                        return '\n'.join(system_lines).strip()
                        
                    except Exception as e:
                        logger.error(f"Failed to extract system message: {e}")
                        return f"Error extracting system message: {e}"
                
                @kernel_function(
                    description="Validate document content for assessment",
                    name="validate_documents"
                )
                def validate_documents(self, documents: str) -> str:
                    """Validate document content using Prompty-based rules"""
                    try:
                        # Basic validation logic
                        if not documents or len(documents.strip()) < 10:
                            return "Invalid: Documents are too short or empty"
                        
                        # Check for required patterns
                        required_patterns = ['Form S', 'Form H', 'Form A']
                        found_patterns = [pattern for pattern in required_patterns if pattern in documents]
                        
                        if not found_patterns:
                            return f"Warning: No standard forms detected. Expected one of: {required_patterns}"
                        
                        return f"Valid: Found {len(found_patterns)} standard forms: {found_patterns}"
                        
                    except Exception as e:
                        logger.error(f"Document validation failed: {e}")
                        return f"Error validating documents: {e}"
            
            # Configure execution settings for Prompty functions
            execution_settings = PromptExecutionSettings(
                service_id="azure_openai_chat",
                max_tokens=4000,
                temperature=0.3,
                top_p=0.9
            )
            
            # Create and register the Prompty plugin
            prompty_plugin = PromptyPlugin(
                self.prompty_template, 
                self.chat_service, 
                execution_settings,
                endpoint=self.endpoint,
                api_key=self.api_key
            )
            self.kernel.add_plugin(prompty_plugin, plugin_name="prompty")
            
            logger.info("Successfully registered Prompty-based functions with Semantic Kernel")
            
        except ImportError as e:
            logger.warning(f"Could not import Prompty: {e}")
        except Exception as e:
            logger.error(f"Failed to create Prompty functions: {e}")
    
    def is_configured(self) -> bool:
        """Check if the client is properly configured"""
        return (self.kernel is not None and 
                self.endpoint is not None and 
                self.api_key is not None)
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get detailed configuration status for debugging with enhanced information"""
        return {
            "semantic_kernel": {
                "available": SK_AVAILABLE,
                "version": SK_VERSION,
                "kernel_initialized": self.kernel is not None,
                "services_count": len(self.kernel.services) if self.kernel else 0,
                "plugins_available": list(self.kernel.plugins.keys()) if self.kernel else [],
                "functions_count": {
                    plugin_name: len(plugin.functions) 
                    for plugin_name, plugin in (self.kernel.plugins.items() if self.kernel else {}).items()
                }
            },
            "prompty": {
                "available": PROMPTY_AVAILABLE,
                "version": PROMPTY_VERSION,
                "template_loaded": hasattr(self, 'prompty_template') and self.prompty_template is not None,
                "executor_configured": hasattr(self, 'prompty_executor') and self.prompty_executor is not None
            },
            "azure_openai": {
                "endpoint_set": bool(self.endpoint),
                "api_key_set": bool(self.api_key),
                "model_name": self.model_name,
                "chat_service_initialized": self.chat_service is not None
            },
            "environment_variables": {
                "LLM_ENDPOINT": "✅ Set" if os.getenv("LLM_ENDPOINT") else "❌ Not set",
                "LLM_API_KEY": "✅ Set" if os.getenv("LLM_API_KEY") else "❌ Not set", 
                "LLM_MODEL_NAME": f"✅ Set to '{os.getenv('LLM_MODEL_NAME', 'gpt-4')}'" if os.getenv("LLM_MODEL_NAME") else "⚠️  Using default 'gpt-4'",
                "USE_OPENAI_CLIENT": f"✅ Set to '{os.getenv('USE_OPENAI_CLIENT', 'true')}'" if os.getenv("USE_OPENAI_CLIENT") else "⚠️  Using default 'true'"
            },
            "helper_modules": {
                "document_extractor": self.document_extractor is not None,
                "prompt_generator": self.prompt_generator is not None
            },
            "overall_health": self._get_health_status()
        }
    
    def _get_health_status(self) -> str:
        """Get overall health status of the integration"""
        if not SK_AVAILABLE:
            return "❌ Critical: Semantic Kernel not available"
        if not self.kernel:
            return "❌ Critical: Kernel not initialized"
        if not self.endpoint or not self.api_key:
            return "❌ Critical: Azure OpenAI credentials missing"
        if not PROMPTY_AVAILABLE:
            return "⚠️  Warning: Prompty not available (limited functionality)"
        if "prompty" not in (self.kernel.plugins.keys() if self.kernel else []):
            return "⚠️  Warning: Prompty plugin not loaded"
        return "✅ Healthy: All components operational"
    
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
    
    async def generate_assessment_report(self, assessment_request: Any, use_prompty_plugin: bool = True) -> Dict[str, Any]:
        """Generate a neurodevelopmental assessment report using Semantic Kernel with performance monitoring
        
        Args:
            assessment_request: The assessment request containing documents
            use_prompty_plugin: Whether to use the Prompty plugin (default: True)
        """
        start_time = datetime.now()
        
        if not self.is_configured():
            return {
                "success": False,
                "error": "Semantic Kernel LLM client not configured properly",
                "report": None,
                "performance": {
                    "total_time_ms": 0,
                    "status": "configuration_error"
                }
            }
        
        try:
            # Preprocess documents to check for issues
            preprocess_start = datetime.now()
            processed_documents = await self._preprocess_documents(assessment_request.documents)
            preprocess_time = (datetime.now() - preprocess_start).total_seconds() * 1000
            
            # Try to use Prompty plugin first if available and requested
            generation_start = datetime.now()
            if use_prompty_plugin and self.kernel and "prompty" in self.kernel.plugins:
                try:
                    report_content = await self._generate_report_with_prompty_plugin(processed_documents)
                    method_used = "prompty_plugin"
                except Exception as e:
                    logger.warning(f"Prompty plugin failed, falling back to traditional method: {e}")
                    report_content = await self._generate_report_traditional(processed_documents)
                    method_used = "traditional_fallback"
            else:
                # Use traditional method
                report_content = await self._generate_report_traditional(processed_documents)
                method_used = "traditional"
            
            generation_time = (datetime.now() - generation_start).total_seconds() * 1000
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Try to get token usage from the result (if available)
            tokens_used = None
            if hasattr(report_content, 'metadata') and hasattr(report_content.metadata, 'get'):
                usage_info = report_content.metadata.get('usage', {})
                if usage_info and hasattr(usage_info, 'get'):
                    tokens_used = usage_info.get('total_tokens')
            
            # Extract content if it's a result object
            if hasattr(report_content, 'content'):
                report_text = str(report_content.content)
            else:
                report_text = str(report_content)
            
            # Create response with enhanced metadata
            response = {
                "success": True,
                "report": report_text,
                "metadata": {
                    "session_id": assessment_request.session_id,
                    "generation_timestamp": datetime.now().isoformat(),
                    "model_used": self.model_name,
                    "method_used": method_used,
                    "documents_processed": len(assessment_request.documents),
                    "required_documents": len([d for d in assessment_request.documents if d.is_required]),
                    "optional_documents": len([d for d in assessment_request.documents if not d.is_required]),
                    "tokens_used": tokens_used,
                    "using_semantic_kernel": True,
                    "kernel_plugins": list(self.kernel.plugins.keys()) if self.kernel else [],
                    "prompty_plugin_available": "prompty" in (self.kernel.plugins.keys() if self.kernel else [])
                },
                "performance": {
                    "total_time_ms": round(total_time, 2),
                    "preprocessing_time_ms": round(preprocess_time, 2),
                    "generation_time_ms": round(generation_time, 2),
                    "tokens_per_second": round(tokens_used / (generation_time / 1000), 2) if tokens_used and generation_time > 0 else None,
                    "characters_generated": len(report_text),
                    "method_used": method_used,
                    "status": "success"
                },
                "error": None
            }
            
            logger.info(f"✅ Successfully generated assessment report using Semantic Kernel ({method_used}) for session {assessment_request.session_id} in {total_time:.2f}ms")
            return response
            
        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"❌ Error generating assessment report with Semantic Kernel: {e}")
            return {
                "success": False,
                "error": str(e),
                "report": None,
                "metadata": {
                    "session_id": assessment_request.session_id,
                    "generation_timestamp": datetime.now().isoformat(),
                    "error_timestamp": datetime.now().isoformat(),
                    "using_semantic_kernel": True
                },
                "performance": {
                    "total_time_ms": round(total_time, 2),
                    "status": "error",
                    "error": str(e)
                }
            }
    
    async def _generate_report_with_prompty_plugin(self, documents: List[Any]) -> str:
        """Generate report using the Prompty plugin"""
        # Format documents for the prompty plugin
        document_content = "\n\n---DOCUMENT---\n\n".join([
            f"DOCUMENT: {doc.filename}\nCONTENT:\n{doc.content}"
            for doc in documents
        ])
        
        # Get the prompty plugin
        prompty_plugin = self.kernel.plugins["prompty"]
        generate_function = prompty_plugin["generate_assessment_report"]
        
        # Invoke the Prompty-based function
        result = await generate_function.invoke(self.kernel, documents=document_content)
        
        return str(result.value) if hasattr(result, 'value') else str(result)
    
    async def _generate_report_traditional(self, documents: List[Any]) -> str:
        """Generate report using traditional prompt generator method"""
        if not self.prompt_generator:
            raise ValueError("Prompt generator not available")
        
        # Create the assessment prompt
        prompt = self.prompt_generator.create_assessment_prompt(documents)
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
        
        return str(result.content) if hasattr(result, 'content') else str(result)
    
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
    
    async def test_prompty_functions(self) -> Dict[str, Any]:
        """Test the Prompty-based functions for demonstration purposes"""
        if not self.kernel or "prompty" not in self.kernel.plugins:
            return {
                "success": False,
                "error": "Prompty plugin not available",
                "results": {}
            }
        
        try:
            prompty_plugin = self.kernel.plugins["prompty"]
            test_results = {}
            
            # Test system message extraction
            if "get_system_message" in prompty_plugin.functions:
                system_msg_function = prompty_plugin["get_system_message"]
                system_result = await system_msg_function.invoke(self.kernel)
                test_results["system_message"] = {
                    "success": True,
                    "content": str(system_result.value) if hasattr(system_result, 'value') else str(system_result)
                }
            
            # Test document validation
            if "validate_documents" in prompty_plugin.functions:
                validate_function = prompty_plugin["validate_documents"]
                test_docs = "REQUIRED - Form S (test.pdf): Test content for validation."
                validate_result = await validate_function.invoke(self.kernel, documents=test_docs)
                test_results["document_validation"] = {
                    "success": True,
                    "content": str(validate_result.value) if hasattr(validate_result, 'value') else str(validate_result)
                }
            
            return {
                "success": True,
                "results": test_results,
                "available_functions": list(prompty_plugin.functions.keys())
            }
            
        except Exception as e:
            logger.error(f"Error testing Prompty functions: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": {}
            }
    
    async def invoke_prompty_function(self, function_name: str, **kwargs) -> Dict[str, Any]:
        """Invoke a specific Prompty function by name
        
        Args:
            function_name: Name of the function to invoke
            **kwargs: Arguments to pass to the function
        """
        if not self.kernel or "prompty" not in self.kernel.plugins:
            return {
                "success": False,
                "error": "Prompty plugin not available",
                "result": None
            }
        
        try:
            prompty_plugin = self.kernel.plugins["prompty"]
            
            if function_name not in prompty_plugin.functions:
                return {
                    "success": False,
                    "error": f"Function '{function_name}' not found",
                    "available_functions": list(prompty_plugin.functions.keys()),
                    "result": None
                }
            
            function = prompty_plugin[function_name]
            result = await function.invoke(self.kernel, **kwargs)
            
            return {
                "success": True,
                "result": str(result.value) if hasattr(result, 'value') else str(result),
                "function_name": function_name,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error invoking Prompty function '{function_name}': {e}")
            return {
                "success": False,
                "error": str(e),
                "result": None
            }


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