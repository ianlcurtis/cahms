import os
import streamlit as st
from dotenv import load_dotenv
import asyncio
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

title = os.getenv("TITLE", "CAHMS Neurodevelopmental Assessment Tool")
logo = os.getenv("LOGO_URL", "images/azure_logo.png")

# Set page configuration FIRST - before any other Streamlit commands
st.set_page_config(page_title=title, page_icon=None, layout="wide")

# Import the Azure LLM client and assessment prompt functionality
try:
    from azure_llm_client_sk import AzureLLMClientSemanticKernel as LLMClient
    #from azure_llm_client_api import AzureLLMClient as LLMClient
    from document_extractor import DocumentExtractor, process_assessment_documents
    from assessment_prompt import AssessmentPromptGenerator
except ImportError:
    st.error("Azure LLM client not found. Make sure azure_llm_client_*.py is in the src folder.")
    LLMClient = None
    process_assessment_documents = None
    DocumentExtractor = None
    AssessmentPromptGenerator = None

# Configure mandatory uploads
MANDATORY_UPLOADS = {
    "form_s": os.getenv("MANDATORY_FORM_S", "true").lower() == "true",
    "form_h": os.getenv("MANDATORY_FORM_H", "false").lower() == "true", 
    "form_a": os.getenv("MANDATORY_FORM_A", "false").lower() == "true",
    "cahms_initial": os.getenv("MANDATORY_CAHMS_INITIAL", "false").lower() == "true",
    "neuro_dev_history": os.getenv("MANDATORY_NEURO_DEV_HISTORY", "false").lower() == "true",
    "formulation_document": os.getenv("MANDATORY_FORMULATION_DOCUMENT", "false").lower() == "true",
    "school_observation": os.getenv("MANDATORY_SCHOOL_OBSERVATION", "false").lower() == "true",
    "supporting_information": os.getenv("MANDATORY_SUPPORTING_INFO", "false").lower() == "true"
}

# File configurations
FILE_CONFIGS = {
    "form_s": {
        "display_name": "Form S",
        "description": "The purpose of Form S is to gather school information for a CAMHS (Child and Adolescent Mental Health Services) Neurodevelopmental Assessment. It aims to understand the strategies and support provided to the child or young person, and why this support is necessary, to better comprehend their neurodevelopmental needs."
    },
    "form_h": {
        "display_name": "Form H",
        "description": "The purpose of Form H is to gather school information for a CAMHS (Child and Adolescent Mental Health Services) Neurodevelopmental Assessment. It aims to understand the young person's attention, concentration, and hyperactivity levels."
    },
    "form_a": {
        "display_name": "Form A",
        "description": "The purpose of Form A is to gather school information for a CAMHS (Child and Adolescent Mental Health Services) Neurodevelopmental Assessment. It aims to understand the young person's social interactions, communication skills, and any restricted or repetitive behaviours."
    },
    "cahms_initial": {
        "display_name": "CAHMS Initial Assessment Document",
        "description": "The purpose of this form is to document the initial appointment details for a young person with the CAMHS Neurodevelopmental Team. It includes information about the presenting complaint, family history, patient history, education, developmental history, and clinical observations."
    },
    "neuro_dev_history": {
        "display_name": "Neuro Dev History",
        "description": "The purpose of this form is to gather comprehensive information about a child's developmental history, family context, and environment. It aims to understand the main challenges the child faces at school and home, the family's mental and physical health history, and the child's early years."
    },
    "formulation_document": {
        "display_name": "Formulation Document",
        "description": "The formulation document provides a comprehensive clinical summary and analysis of the assessment findings. It synthesizes information from all sources to develop a clear understanding of the young person's neurodevelopmental profile, strengths, challenges, and recommended interventions."
    },
    "school_observation": {
        "display_name": "School Observation",
        "description": "This document provides additional insights from direct school observations of the child or young person in their educational environment. It can include observations of behavior, interactions, learning patterns, and social engagement."
    },
    "supporting_information": {
        "display_name": "Supporting Information",
        "description": "This section allows for the upload of any additional supporting documentation that may be relevant to the neurodevelopmental assessment. This could include previous reports, specialist assessments, or other relevant clinical information."
    }
}

def initialize_session_state():
    """Initialize all session state variables"""
    # Result states
    for key in ["output", "report_metadata", "generation_in_progress"]:
        if key not in st.session_state:
            st.session_state[key] = "" if key == "output" else None if key == "report_metadata" else False
    
    # File upload states
    for key in FILE_CONFIGS.keys():
        if key not in st.session_state:
            st.session_state[key] = None

def create_file_upload_container(file_key, column):
    """Create a file upload container for a specific file type"""
    config = FILE_CONFIGS[file_key]
    is_mandatory = MANDATORY_UPLOADS[file_key]
    optional_text = "" if is_mandatory else " *(Optional)*"
    
    with column:
        with st.container(border=True):
            st.markdown(f"### {config['display_name']}{optional_text}")
            st.markdown(config['description'])
            
            upload_label = f"Upload {config['display_name']}" + ("" if is_mandatory else " (Optional)")
            help_text = f"Upload {config['display_name']} document" + ("" if is_mandatory else " (optional)")
            
            uploaded_file = st.file_uploader(
                upload_label,
                type=['pdf', 'doc', 'docx', 'txt'],
                key=f"{file_key}_uploader",
                help=help_text
            )
            
            if uploaded_file:
                st.session_state[file_key] = uploaded_file

# Initialize session state
initialize_session_state()

# Display logo and title
col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists(logo):
        st.image(logo, width=100)
with col2:
    st.title(title)

st.markdown("---")

# File upload section
st.header("Document Upload")
st.markdown("Please upload the required documents for neurodevelopmental assessment:")

# Create file upload fields in two columns
col1, col2 = st.columns(2)

# Left column files
left_column_files = ["form_s", "form_a", "neuro_dev_history", "formulation_document"]
for file_key in left_column_files:
    create_file_upload_container(file_key, col1)

# Right column files  
right_column_files = ["form_h", "cahms_initial", "school_observation", "supporting_information"]
for file_key in right_column_files:
    create_file_upload_container(file_key, col2)

# Display upload status
def display_upload_status():
    """Display the current upload status for all files"""
    all_files = [(key, config["display_name"], st.session_state[key]) 
                 for key, config in FILE_CONFIGS.items()]
    
    mandatory_files = [(key, name, file) for key, name, file in all_files if MANDATORY_UPLOADS[key]]
    optional_files = [(key, name, file) for key, name, file in all_files if not MANDATORY_UPLOADS[key]]
    
    st.markdown("### Upload Status")
    
    if mandatory_files:
        st.markdown("**Required Documents:**")
        mandatory_cols = st.columns(len(mandatory_files))
        for i, (key, name, file) in enumerate(mandatory_files):
            with mandatory_cols[i]:
                if file:
                    st.success(f"✅ {name}")
                else:
                    st.error(f"❌ {name}")
    
    if optional_files:
        st.markdown("**Optional Documents:**")
        optional_cols = st.columns(len(optional_files))
        for i, (key, name, file) in enumerate(optional_files):
            with optional_cols[i]:
                if file:
                    st.success(f"✅ {name}")
                else:
                    st.info(f"⚪ {name} (Optional)")

display_upload_status()

st.markdown("---")

def create_assessment_prompt_and_system_message(documents):
    """
    Create assessment prompt and system message from processed documents
    
    Args:
        documents: List of processed documents
        
    Returns:
        Tuple of (prompt, system_message)
    """
    # Initialize prompt generator
    if AssessmentPromptGenerator:
        prompt_generator = AssessmentPromptGenerator()
        try:
            prompt = prompt_generator.create_assessment_prompt(documents)
            system_message = prompt_generator.create_system_message()
            return prompt, system_message
        except Exception as e:
            st.warning(f"Could not load prompt template: {str(e)}. Using fallback prompt.")
    
    # Fallback prompt generation if AssessmentPromptGenerator is not available
    document_content = ""
    for doc in documents:
        status = "REQUIRED" if doc.is_required else "OPTIONAL"
        document_content += f"\n{status} - {doc.document_type} ({doc.filename}):\n"
        document_content += f"{doc.content}\n"
        document_content += "-" * 80 + "\n"
    
    fallback_prompt = f"""You are a specialist clinician working with the CAMHS (Child and Adolescent Mental Health Services) Neurodevelopmental Team.

Please analyze the provided assessment documents and generate a comprehensive neurodevelopmental assessment report.

AVAILABLE DOCUMENTS:
{document_content}

Generate the assessment report now:
"""
    
    fallback_system_message = "You are an expert clinician specializing in neurodevelopmental assessments for children and adolescents."
    
    return fallback_prompt, fallback_system_message

# Generate button
async def generate_report_async():
    """Async function to handle report generation with LLM"""
    if not LLMClient:
        st.error("LLM client not available. Please check the azure_llm_client_*.py file.")
        return
    
    try:
        # Initialize LLM client
        llm_client = LLMClient()
        if not llm_client.is_configured():
            st.error("LLM client not configured. Please check your .env file with LLM_ENDPOINT and LLM_API_KEY.")
            return
        
        # Prepare uploaded files dictionary
        uploaded_files_dict = {key: st.session_state[key] 
                             for key in FILE_CONFIGS.keys() 
                             if st.session_state[key] is not None}
        
        # Process documents
        session_id = str(uuid.uuid4())
        assessment_request = await process_assessment_documents(uploaded_files_dict, session_id, MANDATORY_UPLOADS)
        
        if not assessment_request.documents:
            st.error("No documents could be processed. Please check your uploaded files.")
            return
        
        # Validate documents
        if DocumentExtractor:
            document_extractor = DocumentExtractor()
            validation_result = await document_extractor.validate_documents(assessment_request.documents)
            if not validation_result["valid"]:
                st.error(f"Document validation failed: {'; '.join(validation_result['errors'])}")
                return
            
            for warning in validation_result.get("warnings", []):
                st.warning(warning)
        else:
            st.warning("Document validation not available - DocumentExtractor not imported")
        
        # Generate prompt and system message locally
        prompt, system_message = create_assessment_prompt_and_system_message(assessment_request.documents)
        
        # Create a simple request object with the prepared prompt
        class PromptRequest:
            def __init__(self, prompt, system_message, documents):
                self.prompt = prompt
                self.system_message = system_message
                self.documents = documents
                self.session_id = assessment_request.session_id
                self.request_timestamp = assessment_request.request_timestamp
        
        prompt_request = PromptRequest(prompt, system_message, assessment_request.documents)
        
        # Generate the report
        start_time = datetime.now()
        with st.spinner("Generating neurodevelopmental assessment report... This may take a few minutes."):
            result = await llm_client.generate_response(prompt_request)
        end_time = datetime.now()
        
        # Calculate duration
        duration = end_time - start_time
        duration_seconds = duration.total_seconds()
        
        if result["success"]:
            # Add duration to metadata
            if result.get("metadata"):
                result["metadata"]["call_duration_seconds"] = duration_seconds
                result["metadata"]["call_duration_formatted"] = f"{int(duration_seconds // 60)}m {int(duration_seconds % 60)}s"
            
            st.session_state.output = result["report"]
            st.session_state.report_metadata = result["metadata"]
            st.success("Assessment report generated successfully!")
        else:
            st.error(f"Failed to generate report: {result['error']}")
            st.session_state.output = ""
            st.session_state.report_metadata = None
            
    except Exception as e:
        st.error(f"Error during report generation: {str(e)}")
        st.session_state.output = ""
        st.session_state.report_metadata = None
    finally:
        st.session_state.generation_in_progress = False

def generate_report():
    """Function to handle report generation"""
    # Check if all required files are uploaded
    mandatory_files = [(key, config["display_name"], st.session_state[key]) 
                      for key, config in FILE_CONFIGS.items() if MANDATORY_UPLOADS[key]]
    required_files = [file for _, _, file in mandatory_files]
    all_required_uploaded = all(required_files)
    
    if not all_required_uploaded:
        st.error("Please upload all required documents before generating the report.")
        return
    
    if st.session_state.generation_in_progress:
        st.warning("Report generation already in progress...")
        return
    
    # Set generation in progress
    st.session_state.generation_in_progress = True
    
    # Run async function
    asyncio.run(generate_report_async())

# Center the generate button
col1, col2, col3 = st.columns([2, 3, 2])
with col2:
    if st.button("Generate Neurodevelopmental Assessment Report", 
                 type="primary", 
                 use_container_width=True,
                 on_click=generate_report):
        pass

# Display output if available
def display_report():
    """Display the generated report with download and action buttons"""
    if not st.session_state.output:
        return
        
    st.markdown("---")
    st.header("Generated Assessment Report")
    
    # Display metadata if available
    if st.session_state.report_metadata:
        with st.expander("Report Metadata", expanded=False):
            metadata = st.session_state.report_metadata
            
            # Compact metadata display in 3 columns with smaller font
            st.markdown(
                """
                <style>
                .compact-metadata {
                    font-size: 0.8em;
                    line-height: 1.2;
                }
                .compact-metadata .metric-label {
                    font-weight: bold;
                    color: #666;
                    margin-bottom: 2px;
                }
                .compact-metadata .metric-value {
                    font-size: 1.1em;
                    color: #333;
                    margin-bottom: 8px;
                }
                </style>
                """, 
                unsafe_allow_html=True
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                docs_processed = str(metadata.get("documents_processed", "N/A"))
                required_docs = str(metadata.get("required_documents", "N/A"))
                optional_docs = str(metadata.get("optional_documents", "N/A"))
                
                st.markdown(
                    f"""<div class="compact-metadata">
                        <div class="metric-label">Documents Processed</div>
                        <div class="metric-value">{docs_processed}</div>
                        <div class="metric-label">Required Documents</div>
                        <div class="metric-value">{required_docs}</div>
                        <div class="metric-label">Optional Documents</div>
                        <div class="metric-value">{optional_docs}</div>
                    </div>""", 
                    unsafe_allow_html=True
                )
            
            with col2:
                duration_formatted = metadata.get("call_duration_formatted", "N/A")
                duration_seconds = metadata.get("call_duration_seconds", "N/A")
                duration_display = str(duration_formatted)
                if duration_seconds != "N/A":
                    duration_display += f" ({duration_seconds:.1f}s)"
                
                model_used = str(metadata.get("model_used", "N/A"))
                total_tokens = str(metadata.get("total_tokens", "N/A"))
                
                st.markdown(
                    f"""<div class="compact-metadata">
                        <div class="metric-label">Model Used</div>
                        <div class="metric-value">{model_used}</div>
                        <div class="metric-label">Generation Time</div>
                        <div class="metric-value">{duration_display}</div>
                        <div class="metric-label">Total Tokens</div>
                        <div class="metric-value">{total_tokens}</div>
                    </div>""", 
                    unsafe_allow_html=True
                )
            
            with col3:
                prompt_tokens = str(metadata.get("prompt_tokens", "N/A"))
                completion_tokens = str(metadata.get("completion_tokens", "N/A"))
                generation_time = metadata.get('generation_timestamp', 'N/A')
                
                if generation_time != 'N/A':
                    try:
                        # Parse and format the timestamp for better display
                        dt = datetime.fromisoformat(generation_time.replace('Z', '+00:00'))
                        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        formatted_time = str(generation_time)
                else:
                    formatted_time = "N/A"
                
                st.markdown(
                    f"""<div class="compact-metadata">
                        <div class="metric-label">Prompt Tokens</div>
                        <div class="metric-value">{prompt_tokens}</div>
                        <div class="metric-label">Response Tokens</div>
                        <div class="metric-value">{completion_tokens}</div>
                        <div class="metric-label">Generated</div>
                        <div class="metric-value">{formatted_time}</div>
                    </div>""", 
                    unsafe_allow_html=True
                )
    
    # Create two columns for the report display and action buttons
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Display the report in a scrollable container
        st.markdown(
            f"""
            <div style="
                max-height: 600px; 
                overflow-y: auto; 
                padding: 20px; 
                border: 1px solid #ddd; 
                border-radius: 5px; 
                background-color: #f9f9f9;
                font-family: 'Arial', sans-serif;
                line-height: 1.6;
            ">
                {st.session_state.output.replace('\n', '<br>')}
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        # Download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CAHMS_Assessment_Report_{timestamp}.txt"
        
        st.download_button(
            label="Download Report",
            data=st.session_state.output,
            file_name=filename,
            mime="text/plain",
            type="secondary",
            use_container_width=True,
            help="Download the assessment report as a text file"
        )
        
        # Additional options
        st.markdown("---")
        st.markdown("**Additional Options:**")
        
        # Copy to clipboard
        if st.button("Copy to Clipboard", use_container_width=True):
            st.components.v1.html(
                f"""
                <script>
                navigator.clipboard.writeText(`{st.session_state.output.replace('`', '\\`')}`).then(function() {{
                    console.log('Report copied to clipboard');
                }});
                </script>
                """,
                height=0
            )
            st.success("Report copied to clipboard!")
        
        # Print view
        if st.button("Print View", use_container_width=True):
            current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            st.components.v1.html(
                f"""
                <script>
                var printWindow = window.open('', '', 'height=600,width=800');
                printWindow.document.write('<html><head><title>CAHMS Assessment Report</title>');
                printWindow.document.write('<style>body{{font-family: Arial, sans-serif; line-height: 1.6; margin: 40px;}}</style>');
                printWindow.document.write('</head><body>');
                printWindow.document.write('<h1>CAHMS Neurodevelopmental Assessment Report</h1>');
                printWindow.document.write('<p><strong>Generated:</strong> {current_time}</p>');
                printWindow.document.write('<hr>');
                printWindow.document.write(`{st.session_state.output.replace('\n', '<br>').replace('`', '\\`')}`);
                printWindow.document.write('</body></html>');
                printWindow.document.close();
                printWindow.print();
                </script>
                """,
                unsafe_allow_html=True
            )

display_report()


