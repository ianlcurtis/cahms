# CAHMS Neurodevelopmental Assessment Tool (.NET)

An ASP.NET Core MVC web application for generating neurodevelopmental assessment reports for Child and Adolescent Mental Health Services (CAHMS). This tool uses Azure OpenAI services and template-based prompts to analyze uploaded assessment documents and generate professional clinical reports.

![CAHMS UI Screenshot](../images/cahms_ui.png)

## Features

- **Web-based Interface**: ASP.NET Core MVC application with responsive design
- **Multi-document Processing**: Supports PDF, DOCX, and TXT documents for various assessment forms
- **Intelligent Analysis**: Uses Azure OpenAI via Semantic Kernel to analyze and synthesize assessment information
- **Template-based Prompts**: Uses versioned text templates for structured prompt engineering
- **Configurable Requirements**: Customizable mandatory/optional document requirements via configuration
- **Professional Reports**: Generates comprehensive neurodevelopmental assessment reports
- **File Upload Management**: Secure file handling with size and type validation
- **Environment Configuration**: Supports both appsettings.json and environment variable configuration

## Supported Document Types

The application accepts PDF, DOCX, and TXT files for the following assessment forms:

- **Form S**: School-based assessment documents
- **Form H**: Hyperactivity and attention assessment
- **Form A**: Social interaction and communication assessment
- **CAHMS Initial Assessment**: Initial clinical assessment documents
- **Neurodevelopmental History**: Developmental history documentation
- **Formulation Document**: Clinical formulation reports
- **School Observation**: Classroom observation reports
- **Supporting Information**: Additional relevant documents

## Prerequisites

- .NET 8.0 SDK or later
- Azure OpenAI service endpoint and API key
- Visual Studio 2022 or VS Code with C# extension
- Azure CLI (for deployment)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cahms/dotnet
```

2. Restore dependencies:
```bash
dotnet restore
```

3. Create a `.env` file in the project root (two levels up from the .csproj file) with the following configuration:
```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=your-azure-openai-endpoint
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_MAX_TOKENS=4000
AZURE_OPENAI_TEMPERATURE=0.7
AZURE_OPENAI_MAX_RETRIES=3
AZURE_OPENAI_TIMEOUT_SECONDS=120

# File Upload Configuration
FILE_UPLOAD_MAX_FILE_SIZE=10485760
FILE_UPLOAD_MAX_TOTAL_SIZE=52428800
FILE_UPLOAD_ALLOWED_EXTENSIONS=.pdf,.docx,.txt
FILE_UPLOAD_PATH=uploads

# Application Configuration
APPLICATION_TITLE=CAHMS Neurodevelopmental Assessment Tool
APPLICATION_LOGO_URL=images/azure_logo.png

# Assessment Configuration
ASSESSMENT_SYSTEM_MESSAGE_PATH=../../prompts/system_message.txt
ASSESSMENT_PROMPT_TEMPLATE_PATH=../../prompts/assessment_prompt_template.txt
```

4. Alternatively, configure settings in `appsettings.json` or `appsettings.Development.json`.

## Running the Application

### Local Development

1. Build the project:
```bash
dotnet build
```

2. Run the application:
```bash
dotnet run --project src/CahmsAssessmentTool
```

3. Navigate to `https://localhost:5001` or `http://localhost:5000` in your browser.

### Using VS Code Tasks

The project includes predefined tasks that can be run from VS Code:

- **Build**: `Ctrl+Shift+P` → "Tasks: Run Task" → "build"
- **Watch**: `Ctrl+Shift+P` → "Tasks: Run Task" → "watch" (for hot reload during development)
- **Publish**: `Ctrl+Shift+P` → "Tasks: Run Task" → "publish"

### Development Container

The project includes a development container configuration in `.devcontainer/` for consistent development environments. Use VS Code with the Dev Containers extension to automatically set up the development environment.

### Azure Deployment

1. Build for production:
```bash
dotnet publish -c Release -o ./publish
```

2. Deploy using Azure infrastructure templates:
```bash
az login
az group create --name rg-cahms --location uksouth
az deployment group create --resource-group rg-cahms --template-file ../infra/core.bicep --parameters ../infra/test.bicepparam
```

## Project Structure

```
dotnet/
├── src/
│   └── CahmsAssessmentTool/
│       ├── Controllers/
│       │   └── HomeController.cs           # Main MVC controller
│       ├── Models/
│       │   ├── DocumentModels.cs           # Document-related data models
│       │   ├── LLMModels.cs                # Azure OpenAI response models
│       │   ├── SettingsModels.cs           # Configuration models
│       │   └── ViewModels.cs               # MVC view models
│       ├── Services/
│       │   ├── IAssessmentPromptGenerator.cs  # Prompt generation interface
│       │   ├── AssessmentPromptGenerator.cs   # Template-based prompt generation
│       │   ├── IAzureLLMClient.cs             # LLM client interface
│       │   ├── AzureLLMClientSemanticKernel.cs # Semantic Kernel LLM client
│       │   ├── IDocumentExtractor.cs          # Document processing interface
│       │   └── DocumentExtractor.cs           # PDF/DOCX text extraction
│       ├── Views/
│       │   ├── Home/
│       │   │   ├── Index.cshtml            # File upload interface
│       │   │   └── Result.cshtml           # Assessment results display
│       │   └── Shared/
│       │       └── _Layout.cshtml          # Main layout template
│       ├── wwwroot/
│       │   ├── css/
│       │   ├── js/
│       │   ├── images/
│       │   └── lib/                        # Client-side libraries (Bootstrap, etc.)
│       ├── Properties/
│       │   └── launchSettings.json         # Development launch configuration
│       ├── appsettings.json                # Application configuration
│       ├── appsettings.Development.json    # Development-specific configuration
│       ├── CahmsAssessmentTool.csproj      # Project file
│       └── Program.cs                      # Application entry point
├── tests/                                  # Unit and integration tests (future)
└── README.md                              # This file
```

## Configuration

The application supports configuration through multiple sources (in order of precedence):

1. **Environment Variables** (highest priority)
2. **appsettings.Development.json** (in Development environment)
3. **appsettings.json** (base configuration)
4. **.env file** (loaded via DotNetEnv)

### Key Configuration Sections

#### Azure OpenAI Settings
```json
{
  "AzureOpenAI": {
    "Endpoint": "your-endpoint",
    "ApiKey": "your-key",
    "DeploymentName": "gpt-4",
    "ApiVersion": "2024-02-15-preview",
    "MaxTokens": 4000,
    "Temperature": 0.7,
    "MaxRetries": 3,
    "TimeoutSeconds": 120
  }
}
```

#### File Upload Settings
```json
{
  "FileUpload": {
    "MaxFileSize": 10485760,
    "MaxTotalSize": 52428800,
    "AllowedExtensions": [".pdf", ".docx", ".txt"],
    "UploadPath": "uploads",
    "RequiredDocuments": {
      "FormH": false,
      "FormS": false,
      "FormA": false,
      "CahmsInitial": false,
      "NeuroDevHistory": false,
      "FormulationDocument": false,
      "SchoolObservation": false,
      "SupportingInfo": false
    }
  }
}
```

## Usage

1. **Start the Application**: Run `dotnet run` and navigate to the local URL
2. **Upload Documents**: Use the web interface to upload assessment documents in PDF, DOCX, or TXT format
3. **Configure Requirements**: Modify the `RequiredDocuments` settings to specify which document types are mandatory
4. **Generate Report**: Click "Generate Assessment Report" to process documents and create a comprehensive report
5. **Review Output**: The generated report is displayed with options to download or copy the results

## Dependencies

The project uses the following key NuGet packages:

- **Microsoft.AspNetCore.App** (8.0+): ASP.NET Core framework
- **Microsoft.SemanticKernel** (1.65.0): Azure OpenAI integration and orchestration
- **Microsoft.SemanticKernel.Connectors.OpenAI** (1.65.0): OpenAI connector for Semantic Kernel
- **DocumentFormat.OpenXml** (3.3.0): DOCX document processing
- **PdfSharpCore** (1.3.67): PDF document processing
- **DotNetEnv** (3.1.1): Environment variable loading from .env files

## Development

### Adding New Document Types

1. Add the document type to `DocumentType` enum in `DocumentModels.cs`
2. Update the `RequiredDocuments` configuration model in `SettingsModels.cs`
3. Modify the upload interface in `Views/Home/Index.cshtml`
4. Update validation logic in `HomeController.cs`

### Customizing Prompts

The application uses external template files for prompts:
- **System Message**: `../../prompts/system_message.txt`
- **Assessment Template**: `../../prompts/assessment_prompt_template.txt`

Modify these files to customize the AI analysis behavior.

### Adding New LLM Clients

1. Implement the `IAzureLLMClient` interface
2. Register the new client in `Program.cs`
3. Update configuration models if needed

## Testing

```bash
# Run all tests
dotnet test

# Run with coverage
dotnet test --collect:"XPlat Code Coverage"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the existing code style and patterns
4. Add appropriate tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions, please contact the CAHMS Development Team.

## License

This project is part of the CAHMS assessment tool suite and follows the same licensing as the parent project.