# CAHMS Neurodevelopmental Assessment Tool

A Streamlit application for generating neurodevelopmental assessment reports for Child and Adolescent Mental Health Services (CAHMS). This tool uses Azure OpenAI services and template-based prompts to analyze uploaded assessment documents and generate professional clinical reports.

![CAHMS UI Screenshot](images/cahms_ui.png)


## Features

- **Multi-document Processing**: Supports PDF and Word documents for various assessment forms
- **Intelligent Analysis**: Uses Azure OpenAI to analyze and synthesize assessment information
- **Template-based Prompts**: Uses versioned text templates for structured prompt engineering
- **Configurable Requirements**: Customizable mandatory/optional document requirements
- **Professional Reports**: Generates comprehensive neurodevelopmental assessment reports
- **Multiple LLM Clients**: Supports both direct API and Semantic Kernel integration

## Supported Document Types

The application accepts PDF, DOC, DOCX, and TXT files for the following assessment forms:

- **Form S**: School-based assessment documents
- **Form H**: Hyperactivity and attention assessment
- **Form A**: Social interaction and communication assessment
- **CAHMS Initial Assessment**: Initial clinical assessment documents
- **Neurodevelopmental History**: Developmental history documentation
- **Formulation Document**: Clinical formulation reports
- **School Observation**: Classroom observation reports
- **Supporting Information**: Additional relevant documents

## Prerequisites

- Python 3.12+ (3.8+ minimum)
- Azure OpenAI service endpoint and API key
- Azure CLI (for deployment)
- Streamlit

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cahms
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following configuration:
```env
# Application Configuration
TITLE="CAHMS Neurodevelopmental Assessment Tool"
LOGO_URL="images/azure_logo.png"

# Azure OpenAI/LLM Configuration
LLM_ENDPOINT="your-azure-openai-endpoint"
LLM_API_KEY="your-api-key"
LLM_MODEL_NAME="gpt-4"
USE_OPENAI_CLIENT="true"

# Mandatory Document Configuration (optional)
MANDATORY_FORM_S="true"
MANDATORY_FORM_H="false"
MANDATORY_FORM_A="false"
MANDATORY_CAHMS_INITIAL="false"
MANDATORY_NEURO_DEV_HISTORY="false"
MANDATORY_FORMULATION_DOCUMENT="false"
MANDATORY_SCHOOL_OBSERVATION="false"
MANDATORY_SUPPORTING_INFO="false"
```

## Running the Application

### Local Development
```bash
streamlit run src/Home.py
```

### Development Container
The project includes a development container configuration in `.devcontainer/` for consistent development environments. Use VS Code with the Dev Containers extension to automatically set up the development environment.

### Azure Deployment

1. Create a resource group:
```bash
az login
az group create --name rg-cahms --location uksouth
```

2. Deploy the infrastructure:
```bash
az deployment group create --resource-group rg-cahms --template-file infra/core.bicep --parameters infra/test.bicepparam
```

3. Preview deployment changes:
```bash
az deployment group what-if --resource-group rg-cahms --template-file infra/core.bicep --parameters infra/test.bicepparam
```

## Project Structure

```
cahms/
├── src/
│   ├── Home.py                      # Main Streamlit application
│   ├── assessment_prompt.py         # Template-based prompt generation
│   ├── azure_llm_client_api.py      # Azure OpenAI direct API client
│   ├── azure_llm_client_sk.py       # Azure OpenAI Semantic Kernel client
│   └── document_extractor.py        # Document processing utilities
├── prompts/
│   ├── assessment_prompt_template.txt # Assessment prompt template
│   └── system_message.txt           # System message template
├── infra/
│   ├── core.bicep                   # Azure infrastructure template
│   ├── test.bicepparam              # Test environment parameters
│   └── prod.bicepparam              # Production environment parameters
├── images/                          # Static assets and logos
├── .devcontainer/                   # Development container configuration
├── .github/                         # GitHub workflows and configurations
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## Usage

1. **Upload Documents**: Upload assessment documents in PDF or Word format using the file upload interface
2. **Configure Requirements**: Set which documents are mandatory vs optional via environment variables
3. **Generate Report**: Click "Generate Neurodevelopmental Assessment Report" to process documents and create a comprehensive report
4. **Review Output**: The generated report includes metadata, download options, and comprehensive assessment findings
5. **Export Options**: Download as text file, copy to clipboard, or print the report

## Configuration

The application can be customized through environment variables:

- **LLM Configuration**: Set `LLM_ENDPOINT`, `LLM_API_KEY`, and `LLM_MODEL_NAME` for Azure OpenAI
- **Document Requirements**: Control which document types are mandatory using `MANDATORY_*` variables
- **Branding**: Customize title and logo with `TITLE` and `LOGO_URL`
- **Client Selection**: Choose between direct API or Semantic Kernel with `USE_OPENAI_CLIENT`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions, please contact the CAHMS Development Team.