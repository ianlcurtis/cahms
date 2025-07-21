# CAHMS Neurodevelopmental Assessment Tool

A Streamlit application for generating neurodevelopmental assessment reports for Child and Adolescent Mental Health Services (CAHMS). This tool uses Azure OpenAI services and Prompty templates to analyze uploaded assessment documents and generate professional clinical reports.

![CAHMS UI Screenshot](images/cahms_ui.png)


## Features

- **Multi-document Processing**: Supports PDF and Word documents for various assessment forms
- **Intelligent Analysis**: Uses Azure OpenAI to analyze and synthesize assessment information
- **Prompty Integration**: Leverages Prompty templates for structured prompt engineering
- **Configurable Requirements**: Customizable mandatory/optional document requirements
- **Professional Reports**: Generates comprehensive neurodevelopmental assessment reports

## Supported Document Types

- **Form S**: School-based assessment documents
- **Form H**: Hyperactivity and attention assessment
- **Form A**: Social interaction and communication assessment
- **CAHMS Initial Assessment**: Initial clinical assessment documents
- **Neurodevelopmental History**: Developmental history documentation
- **Formulation Document**: Clinical formulation reports
- **School Observation**: Classroom observation reports
- **Supporting Information**: Additional relevant documents

## Prerequisites

- Python 3.8+
- Azure OpenAI service endpoint and API key
- Azure CLI (for deployment)

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
LOGO_URL="images/msft_logo.png"

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT="your-openai-endpoint"
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment-name"
AZURE_OPENAI_API_VERSION="2024-02-01"

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
│   ├── Home.py                     # Main Streamlit application
│   ├── assessment_prompt.py        # Prompty-based prompt generation
│   ├── assessment_report.prompty   # Prompty template for reports
│   ├── azure_llm_client_api.py    # Azure OpenAI client
│   └── document_extractor.py      # Document processing utilities
├── infra/
│   ├── core.bicep                 # Azure infrastructure template
│   ├── test.bicepparam            # Test environment parameters
│   └── prod.bicepparam            # Production environment parameters
├── images/                        # Static assets
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## Usage

1. **Upload Documents**: Use the sidebar to upload assessment documents in PDF or Word format
2. **Configure Requirements**: Set which documents are mandatory vs optional via environment variables
3. **Generate Report**: Click "Generate Assessment Report" to process documents and create a comprehensive report
4. **Review Output**: The generated report includes executive summary, background, findings, formulation, and recommendations

## Configuration

The application can be customized through environment variables:

- **Document Requirements**: Control which document types are mandatory
- **Branding**: Customize title and logo
- **Azure OpenAI**: Configure AI service endpoints and models

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions, please contact the CAHMS Development Team.