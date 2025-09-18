# CAHMS Neurodevelopmental Assessment Tool

> **⚠️ DISCLAIMER: This is proof-of-concept code for demonstration purposes only. This software is not production-ready and comes with no warranties or guarantees. It should not be used in clinical or production environments without proper testing, validation, and security review. The authors accept no responsibility for any consequences arising from its use.**

A comprehensive suite of applications for generating neurodevelopmental assessment reports for Child and Adolescent Mental Health Services (CAHMS). This project provides both Python and .NET implementations of an intelligent document processing system that uses Azure OpenAI to analyze clinical assessment documents and generate professional reports.

![CAHMS UI Screenshot](images/cahms_ui.png)

## Overview

The CAHMS Assessment Tool helps clinicians process multiple assessment documents and generate comprehensive neurodevelopmental evaluation reports. The system supports various document types including school assessments, clinical observations, developmental histories, and supporting documentation. Using Azure OpenAI services with template-based prompts, it analyzes uploaded documents and synthesizes the information into structured clinical reports.

## Key Features

- **Multi-document Processing**: Supports PDF, Word, and text documents across various assessment forms
- **Intelligent Analysis**: Leverages Azure OpenAI to analyze and synthesize assessment information
- **Template-based Prompts**: Uses versioned text templates for consistent and structured prompt engineering
- **Flexible Configuration**: Customizable mandatory/optional document requirements
- **Professional Reports**: Generates comprehensive neurodevelopmental assessment reports
- **Multiple Implementations**: Available as both Python (Streamlit) and .NET (ASP.NET Core) applications

## Supported Document Types

The system processes the following assessment forms:

- **Form S**: School-based assessment documents
- **Form H**: Hyperactivity and attention assessment
- **Form A**: Social interaction and communication assessment
- **CAHMS Initial Assessment**: Initial clinical assessment documents
- **Neurodevelopmental History**: Developmental history documentation
- **Formulation Document**: Clinical formulation reports
- **School Observation**: Classroom observation reports
- **Supporting Information**: Additional relevant documents

## Available Implementations

This project provides two complete implementations to suit different deployment needs and technical preferences:

### 🐍 Python Implementation (Streamlit)
A web application built with Streamlit, ideal for rapid prototyping and data science workflows.

**Key Technologies:**
- Streamlit for web interface
- Azure OpenAI SDK and Semantic Kernel support
- Python-based document processing
- Simple deployment and configuration

📖 **[View Python Documentation](python/README.md)**

### ⚡ .NET Implementation (ASP.NET Core)
A robust web application built with ASP.NET Core MVC, suitable for enterprise deployments.

**Key Technologies:**
- ASP.NET Core MVC framework
- Semantic Kernel for LLM integration
- Enterprise-ready architecture
- Comprehensive configuration options

📖 **[View .NET Documentation](dotnet/README.md)**

## Quick Start

### Prerequisites
- Azure OpenAI service endpoint and API key
- Python 3.8+ (for Python implementation) or .NET 8.0+ (for .NET implementation)
- Azure CLI (for deployment)

### Choose Your Implementation

#### Python/Streamlit
```bash
cd python
pip install -r requirements.txt
# Configure .env file (see python/README.md)
streamlit run src/Home.py
```

#### .NET/ASP.NET Core
```bash
cd dotnet
dotnet restore
# Configure .env file (see dotnet/README.md)
dotnet run --project src/CahmsAssessmentTool
```

## Project Structure

```
cahms/
├── README.md                           # This overview document
├── python/                             # Python/Streamlit implementation
│   ├── README.md                       # Python-specific documentation
│   ├── src/                           # Python source code
│   ├── requirements.txt               # Python dependencies
│   └── tests/                         # Python tests and evaluation
├── dotnet/                            # .NET/ASP.NET Core implementation
│   ├── README.md                      # .NET-specific documentation
│   ├── src/CahmsAssessmentTool/       # .NET source code
│   └── tests/                         # .NET tests
├── prompts/                           # Shared prompt templates
│   ├── assessment_prompt_template.txt # Assessment generation template
│   └── system_message.txt            # LLM system message
├── infra/                             # Azure infrastructure templates
│   ├── core.bicep                     # Main infrastructure template
│   ├── test.bicepparam               # Test environment parameters
│   └── prod.bicepparam               # Production environment parameters
├── images/                            # Shared static assets
└── cahms.sln                         # .NET solution file
```

## Configuration

Both implementations share common configuration patterns:

### Azure OpenAI Settings
- **Endpoint**: Your Azure OpenAI service endpoint
- **API Key**: Authentication key for the service
- **Model**: GPT-4 or compatible model deployment
- **Parameters**: Temperature, max tokens, timeout settings

### Document Requirements
- Configure which document types are mandatory vs optional
- Customize file upload limits and allowed extensions
- Set validation rules for assessment completeness

### Branding and UI
- Customize application title and branding
- Configure logo and styling options
- Set up environment-specific settings

## Azure Deployment

The project includes Azure Bicep templates for infrastructure as code:

```bash
# Create resource group
az group create --name rg-cahms --location uksouth

# Deploy infrastructure
az deployment group create \
  --resource-group rg-cahms \
  --template-file infra/core.bicep \
  --parameters infra/test.bicepparam

# Preview changes
az deployment group what-if \
  --resource-group rg-cahms \
  --template-file infra/core.bicep \
  --parameters infra/test.bicepparam
```

## Usage Workflow

1. **Upload Documents**: Use the web interface to upload assessment documents in PDF, Word, or text format
2. **Configure Requirements**: Set which document types are mandatory for your assessment process
3. **Generate Report**: Process documents using Azure OpenAI to create comprehensive assessment reports
4. **Review and Export**: Review generated reports with options to download, copy, or print

## Development

### Getting Started
1. Clone the repository
2. Choose your preferred implementation (Python or .NET)
3. Follow the specific setup instructions in the respective README
4. Configure Azure OpenAI credentials
5. Start developing!

### Contributing
1. Fork the repository
2. Create a feature branch
3. Follow the coding standards for your chosen implementation
4. Add appropriate tests
5. Submit a pull request

### Development Container
Both implementations support VS Code development containers for consistent development environments. Use the Dev Containers extension to automatically set up your development environment.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, questions, or feature requests, please contact the CAHMS Development Team or create an issue in the project repository.

---

**Choose your preferred implementation:**
- 🐍 **[Python/Streamlit Documentation](python/README.md)** - For rapid development and data science workflows
- ⚡ **[.NET/ASP.NET Core Documentation](dotnet/README.md)** - For enterprise-ready web applications