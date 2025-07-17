import os
import streamlit as st
from dotenv import load_dotenv
import json
import asyncio
import uuid
from datetime import datetime

# Import the Azure LLM client
try:
    from src.azure_llm_client import AzureLLMClient, process_assessment_documents
except ImportError:
    try:
        from azure_llm_client import AzureLLMClient, process_assessment_documents
    except ImportError:
        st.error("Azure LLM client not found. Make sure azure_llm_client.py is in the src folder.")
        AzureLLMClient = None
        process_assessment_documents = None

# Load environment variables
load_dotenv()

title = os.getenv("TITLE", "CAHMS Neurodevelopmental Assessment Tool")
logo = os.getenv("LOGO_URL", "images/msft_logo.png")

feedback_endpoint = os.getenv("FEEDBACK_ENDPOINT")

# Configure which uploads are mandatory
# This can be customized based on organizational requirements
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

# Initialize session state for result
if "output" not in st.session_state:
    st.session_state.output = ""
if "report_metadata" not in st.session_state:
    st.session_state.report_metadata = None
if "generation_in_progress" not in st.session_state:
    st.session_state.generation_in_progress = False

# Initialize session state for uploaded files
if "form_s" not in st.session_state:
    st.session_state.form_s = None
if "form_h" not in st.session_state:
    st.session_state.form_h = None
if "form_a" not in st.session_state:
    st.session_state.form_a = None
if "cahms_initial" not in st.session_state:
    st.session_state.cahms_initial = None
if "neuro_dev_history" not in st.session_state:
    st.session_state.neuro_dev_history = None
if "formulation_document" not in st.session_state:
    st.session_state.formulation_document = None
if "school_observation" not in st.session_state:
    st.session_state.school_observation = None
if "supporting_information" not in st.session_state:
    st.session_state.supporting_information = None

# Set page configuration
st.set_page_config(page_title=title, page_icon=":brain:", layout="wide")

# Display logo and title
col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists(logo):
        st.image(logo, width=100)
with col2:
    st.title(title)

st.markdown("---")

# Add custom CSS for container styling
st.markdown("""
<style>
div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    background-color: #f9f9f9;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# File upload section
st.header("Document Upload")
st.markdown("Please upload the required documents for neurodevelopmental assessment:")

# Create file upload fields
col1, col2 = st.columns(2)

with col1:
    # Form S Container
    with st.container(border=True):
        st.markdown("### Form S")
        st.markdown("The purpose of Form S is to gather school information for a CAMHS (Child and Adolescent Mental Health Services) Neurodevelopmental Assessment. It aims to understand the strategies and support provided to the child or young person, and why this support is necessary, to better comprehend their neurodevelopmental needs.")
        form_s = st.file_uploader(
            "Upload Form S",
            type=['pdf', 'doc', 'docx', 'txt'],
            key="form_s_uploader",
            help="Upload Form S document"
        )
    
    # Form A Container
    with st.container(border=True):
        st.markdown("### Form A")
        st.markdown("The purpose of Form A is to gather school information for a CAMHS (Child and Adolescent Mental Health Services) Neurodevelopmental Assessment. It aims to understand the young person's social interactions, communication skills, and any restricted or repetitive behaviours.")
        form_a = st.file_uploader(
            "Upload Form A",
            type=['pdf', 'doc', 'docx', 'txt'],
            key="form_a_uploader",
            help="Upload Form A document"
        )
    
    # Neuro Dev History Container
    with st.container(border=True):
        st.markdown("### Neuro Dev History")
        st.markdown("The purpose of this form is to gather comprehensive information about a child's developmental history, family context, and environment. It aims to understand the main challenges the child faces at school and home, the family's mental and physical health history, and the child's early years, including pregnancy and early development. This information helps in identifying any developmental concerns and provides a holistic view of the child's upbringing and current situation.")
        neuro_dev_history = st.file_uploader(
            "Upload Neuro Dev History",
            type=['pdf', 'doc', 'docx', 'txt'],
            key="neuro_dev_history_uploader",
            help="Upload Neurodevelopmental History document"
        )
    
    # Formulation Document Container
    with st.container(border=True):
        st.markdown("### Formulation Document")
        st.markdown("The formulation document provides a comprehensive clinical summary and analysis of the assessment findings. It synthesizes information from all sources to develop a clear understanding of the young person's neurodevelopmental profile, strengths, challenges, and recommended interventions or support strategies.")
        formulation_document = st.file_uploader(
            "Upload Formulation Document",
            type=['pdf', 'doc', 'docx', 'txt'],
            key="formulation_document_uploader",
            help="Upload Formulation Document"
        )

with col2:
    # Form H Container
    with st.container(border=True):
        st.markdown("### Form H")
        st.markdown("The purpose of Form H is to gather school information for a CAMHS (Child and Adolescent Mental Health Services) Neurodevelopmental Assessment. It aims to understand the young person's attention, concentration, and hyperactivity levels.")
        form_h = st.file_uploader(
            "Upload Form H",
            type=['pdf', 'doc', 'docx', 'txt'],
            key="form_h_uploader",
            help="Upload Form H document"
        )
    
    # CAHMS Initial Assessment Container
    with st.container(border=True):
        st.markdown("### CAHMS Initial Assessment Document")
        st.markdown("The purpose of this form is to document the initial appointment details for a young person with the CAMHS Neurodevelopmental Team. It includes information about the presenting complaint, family history, patient history, education, developmental history, and clinical observations. The form aims to gather comprehensive information to assess the young person's needs and plan appropriate care and support.")
        cahms_initial = st.file_uploader(
            "Upload CAHMS Initial Assessment Document",
            type=['pdf', 'doc', 'docx', 'txt'],
            key="cahms_initial_uploader",
            help="Upload CAHMS Initial Assessment document"
        )
    
    # School Observation Container (Optional)
    with st.container(border=True):
        mandatory_text = "" if not MANDATORY_UPLOADS["school_observation"] else " **(Required)**"
        optional_text = " *(Optional)*" if not MANDATORY_UPLOADS["school_observation"] else ""
        st.markdown(f"### School Observation{mandatory_text}{optional_text}")
        st.markdown("This document provides additional insights from direct school observations of the child or young person in their educational environment. It can include observations of behavior, interactions, learning patterns, and social engagement within the school setting.")
        upload_label = "Upload School Observation" + ("" if MANDATORY_UPLOADS["school_observation"] else " (Optional)")
        school_observation = st.file_uploader(
            upload_label,
            type=['pdf', 'doc', 'docx', 'txt'],
            key="school_observation_uploader",
            help="Upload School Observation document" + ("" if MANDATORY_UPLOADS["school_observation"] else " (optional)")
        )
    
    # Supporting Information Container (Optional)
    with st.container(border=True):
        mandatory_text = "" if not MANDATORY_UPLOADS["supporting_information"] else " **(Required)**"
        optional_text = " *(Optional)*" if not MANDATORY_UPLOADS["supporting_information"] else ""
        st.markdown(f"### Supporting Information{mandatory_text}{optional_text}")
        st.markdown("This section allows for the upload of any additional supporting documentation that may be relevant to the neurodevelopmental assessment. This could include previous reports, specialist assessments, or other relevant clinical information.")
        upload_label = "Upload Supporting Information" + ("" if MANDATORY_UPLOADS["supporting_information"] else " (Optional)")
        supporting_information = st.file_uploader(
            upload_label,
            type=['pdf', 'doc', 'docx', 'txt'],
            key="supporting_information_uploader",
            help="Upload Supporting Information documents" + ("" if MANDATORY_UPLOADS["supporting_information"] else " (optional)")
        )

# Store uploaded files in session state
if form_s:
    st.session_state.form_s = form_s
if form_h:
    st.session_state.form_h = form_h
if form_a:
    st.session_state.form_a = form_a
if cahms_initial:
    st.session_state.cahms_initial = cahms_initial
if neuro_dev_history:
    st.session_state.neuro_dev_history = neuro_dev_history
if formulation_document:
    st.session_state.formulation_document = formulation_document
if school_observation:
    st.session_state.school_observation = school_observation
if supporting_information:
    st.session_state.supporting_information = supporting_information

# Display upload status
# Separate files into mandatory and optional based on configuration
all_files = [
    ("form_s", "Form S", st.session_state.form_s),
    ("form_h", "Form H", st.session_state.form_h),
    ("form_a", "Form A", st.session_state.form_a),
    ("cahms_initial", "CAHMS Initial Assessment", st.session_state.cahms_initial),
    ("neuro_dev_history", "Neuro Dev History", st.session_state.neuro_dev_history),
    ("formulation_document", "Formulation Document", st.session_state.formulation_document),
    ("school_observation", "School Observation", st.session_state.school_observation),
    ("supporting_information", "Supporting Information", st.session_state.supporting_information)
]

# Filter into mandatory and optional based on configuration
mandatory_files = [(key, name, file) for key, name, file in all_files if MANDATORY_UPLOADS[key]]
optional_files = [(key, name, file) for key, name, file in all_files if not MANDATORY_UPLOADS[key]]

st.markdown("### Upload Status")

if mandatory_files:
    st.markdown("**Required Documents:**")
    # Create columns based on number of mandatory files
    mandatory_cols = st.columns(len(mandatory_files))
    for i, (key, name, file) in enumerate(mandatory_files):
        with mandatory_cols[i]:
            if file:
                st.success(f"✅ {name}")
            else:
                st.error(f"❌ {name}")

if optional_files:
    st.markdown("**Optional Documents:**")
    # Create columns based on number of optional files
    optional_cols = st.columns(len(optional_files))
    for i, (key, name, file) in enumerate(optional_files):
        with optional_cols[i]:
            if file:
                st.success(f"✅ {name}")
            else:
                st.info(f"⚪ {name} (Optional)")

# Update uploaded_files list for validation
uploaded_files = [(name, file) for key, name, file in mandatory_files]

st.markdown("---")

# Generate button
async def generate_report_async():
    """Async function to handle report generation with LLM"""
    if not AzureLLMClient:
        st.error("LLM client not available. Please check the azure_llm_client.py file.")
        return
    
    try:
        # Create session ID
        session_id = str(uuid.uuid4())
        
        # Initialize LLM client
        llm_client = AzureLLMClient()
        
        if not llm_client.is_configured():
            st.error("LLM client not configured. Please check your .env file with LLM_ENDPOINT and LLM_API_KEY.")
            return
        
        # Prepare uploaded files dictionary with mandatory status
        uploaded_files_dict = {}
        for key in ["form_s", "form_h", "form_a", "cahms_initial", "neuro_dev_history", 
                   "formulation_document", "school_observation", "supporting_information"]:
            file_obj = getattr(st.session_state, key)
            if file_obj:
                uploaded_files_dict[key] = file_obj
        
        # Process documents with mandatory configuration
        assessment_request = await process_assessment_documents(uploaded_files_dict, session_id, MANDATORY_UPLOADS)
        
        if not assessment_request.documents:
            st.error("No documents could be processed. Please check your uploaded files.")
            return
        
        # Validate documents
        validation_result = await llm_client.validate_documents(assessment_request.documents)
        
        if not validation_result["valid"]:
            st.error(f"Document validation failed: {'; '.join(validation_result['errors'])}")
            return
        
        if validation_result["warnings"]:
            for warning in validation_result["warnings"]:
                st.warning(warning)
        
        # Generate the report
        with st.spinner("Generating neurodevelopmental assessment report... This may take a few minutes."):
            result = await llm_client.generate_assessment_report(assessment_request)
        
        if result["success"]:
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
    # Check if all required files are uploaded (optional files don't need to be uploaded)
    required_files = [file for _, file in uploaded_files]
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
    if st.button("🧠 Generate Neurodevelopmental Assessment Report", 
                 type="primary", 
                 use_container_width=True,
                 on_click=generate_report):
        pass

# Display output if available
if st.session_state.output:
    st.markdown("---")
    st.header("Generated Assessment Report")
    
    # Display metadata if available
    if st.session_state.report_metadata:
        with st.expander("Report Metadata", expanded=False):
            metadata = st.session_state.report_metadata
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Documents Processed", metadata.get("documents_processed", "N/A"))
                st.metric("Required Documents", metadata.get("required_documents", "N/A"))
            
            with col2:
                st.metric("Optional Documents", metadata.get("optional_documents", "N/A"))
                st.metric("Model Used", metadata.get("model_used", "N/A"))
            
            with col3:
                st.metric("Tokens Used", metadata.get("tokens_used", "N/A"))
                st.text(f"Generated: {metadata.get('generation_timestamp', 'N/A')}")
    
    # Create two columns for the report display and download button
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
        report_content = st.session_state.output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CAHMS_Assessment_Report_{timestamp}.txt"
        
        st.download_button(
            label="📄 Download Report",
            data=report_content,
            file_name=filename,
            mime="text/plain",
            type="secondary",
            use_container_width=True,
            help="Download the assessment report as a text file"
        )
        
        # Additional download options
        st.markdown("---")
        st.markdown("**Additional Options:**")
        
        # Copy to clipboard button (using JavaScript)
        if st.button("📋 Copy to Clipboard", use_container_width=True):
            st.components.v1.html(
                f"""
                <script>
                navigator.clipboard.writeText(`{report_content.replace('`', '\\`')}`).then(function() {{
                    console.log('Report copied to clipboard');
                }});
                </script>
                """,
                height=0
            )
            st.success("Report copied to clipboard!")
        
        # Print-friendly version
        if st.button("🖨️ Print View", use_container_width=True):
            st.markdown(
                f"""
                <script>
                var printWindow = window.open('', '', 'height=600,width=800');
                printWindow.document.write('<html><head><title>CAHMS Assessment Report</title>');
                printWindow.document.write('<style>body{{font-family: Arial, sans-serif; line-height: 1.6; margin: 40px;}}</style>');
                printWindow.document.write('</head><body>');
                printWindow.document.write('<h1>CAHMS Neurodevelopmental Assessment Report</h1>');
                printWindow.document.write('<p><strong>Generated:</strong> {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>');
                printWindow.document.write('<hr>');
                printWindow.document.write(`{report_content.replace('\n', '<br>').replace('`', '\\`')}`);
                printWindow.document.write('</body></html>');
                printWindow.document.close();
                printWindow.print();
                </script>
                """,
                unsafe_allow_html=True
            )


