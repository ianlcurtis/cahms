# CAHMS Groundedness Evaluation

This directory contains the groundedness evaluation framework for the CAHMS neurodevelopmental assessment application. The evaluation uses Azure AI Foundry's evaluation SDK to assess whether generated reports are properly grounded in the provided source documents.

## Overview

The groundedness evaluation ensures that:
- Generated assessment reports accurately reflect the content of source documents
- No information is hallucinated or added beyond what's provided
- Recommendations are based on documented needs and observations
- Clinical assessments maintain high standards of accuracy and reliability

## Structure

```
tests/evaluation/
├── configs/
│   └── evaluation_config.json      # Configuration settings
├── test_documents/
│   ├── case_001/                   # ADHD assessment case
│   │   ├── metadata.json
│   │   ├── form_s.txt              # School referral form
│   │   └── form_h.txt              # Home referral form
│   └── case_002/                   # ASD assessment case
│       ├── metadata.json
│       ├── form_s.txt              # School referral form
│       └── form_a.txt              # Autism assessment form
├── expected_outputs/
│   ├── case_001_expected.txt       # Expected output criteria
│   └── case_002_expected.txt
└── groundedness_evaluator.py      # Main evaluation framework
```

## Setup

### 1. Dependencies

The evaluation requires the Azure AI Evaluation SDK and related packages:

```bash
pip install azure-ai-evaluation azure-identity azure-ai-projects pytest pytest-asyncio
```

These are already included in `requirements.txt`.

### 2. Azure App Registration & Service Principal

**⚠️ REQUIRED**: You must create an Azure App Registration and configure it properly to access Azure AI Foundry services.

#### Step 2.1: Create App Registration

1. **Navigate to Azure Portal** → **Microsoft Entra ID** → **App registrations**
2. **Click "New registration"**
3. **Configure the registration**:
   - **Name**: `CAHMS-GitHub-Actions` (or descriptive name)
   - **Supported account types**: `Accounts in this organizational directory only`
   - **Redirect URI**: Leave blank (service-to-service authentication)

#### Step 2.2: Configure Authentication for GitHub Actions

**For GitHub Actions (Recommended - OIDC):**
1. Go to **Certificates & secrets** → **Federated credentials**
2. Click **Add credential** → **GitHub Actions deploying Azure resources**
3. Configure:
   - **Organization**: `<YOUR ORGANISATION>`
   - **Repository**: `cahms`
   - **Entity type**: `Branch`
   - **GitHub branch name**: `main`
   - **Name**: `GitHub-Actions-Main`
4. Repeat for `develop` branch if needed

**Alternative - Client Secret (Less Secure):**
1. Go to **Certificates & secrets** → **Client secrets**
2. Click **New client secret**
3. Set expiration (12-24 months recommended)
4. **Copy the secret value immediately** (you won't see it again!)

#### Step 2.3: Assign Azure AI Foundry Permissions

The app registration needs access to Azure AI services through **Azure RBAC** (not API permissions):

**For Azure OpenAI:**
1. Navigate to your **Azure OpenAI resource**
2. Go to **Access control (IAM)** → **Add role assignment**
3. Assign role to your app registration:
   - **Role**: `Cognitive Services OpenAI User` (recommended)
   - **Assign access to**: User, group, or service principal
   - **Select**: Search for your app registration name

**For Azure AI Foundry/Projects:**
1. Navigate to your **Azure AI Foundry project**
2. Go to **Access control (IAM)** → **Add role assignment**
3. Assign role to your app registration:
   - **Role**: `Azure AI Developer` or `Contributor`
   - **Select**: Your app registration

**For Resource Group (if broader access needed):**
1. Navigate to your **Resource Group** containing AI services
2. Go to **Access control (IAM)** → **Add role assignment**
3. Assign appropriate role (`Contributor` or specific service roles)

#### Step 2.4: Collect Required Values

After app registration setup, collect these values:

```bash
# From App Registration Overview page
AZURE_CLIENT_ID=your_application_client_id
AZURE_TENANT_ID=your_directory_tenant_id

# From Azure Subscription
AZURE_SUBSCRIPTION_ID=your_azure_subscription_id

# From Azure OpenAI Resource
LLM_ENDPOINT=https://your-openai-resource.openai.azure.com/
LLM_API_KEY=your_openai_api_key
LLM_MODEL_NAME=your_deployment_name  # e.g., "gpt-4"
```

### 3. Environment Variables

Set the following environment variables:

```bash
# Azure OpenAI Configuration (for both generation and evaluation)
LLM_ENDPOINT=your_azure_openai_endpoint
LLM_API_KEY=your_azure_openai_api_key
LLM_MODEL_NAME=gpt-4

# Azure Authentication (for evaluation services)
AZURE_CLIENT_ID=your_service_principal_client_id
AZURE_TENANT_ID=your_azure_tenant_id
AZURE_SUBSCRIPTION_ID=your_azure_subscription_id
```

### 4. Configuration

Edit `configs/evaluation_config.json` to adjust evaluation settings:

- `groundedness_threshold`: Minimum score (1-5) for groundedness evaluation
- `relevance_threshold`: Minimum score (1-5) for relevance evaluation  
- `pass_rate_threshold`: Minimum percentage of cases that must pass (0.0-1.0)

## Usage

### Local Testing

Run the evaluation locally:

```bash
cd tests/evaluation
python groundedness_evaluator.py
```

### CI/CD Integration

The evaluation runs automatically on:
- Pushes to `main` or `develop` branches
- Pull requests to `main`
- Changes to `src/`, `prompts/`, or `tests/evaluation/` directories

### GitHub Workflow

The evaluation is integrated into the deployment pipeline:

1. **Groundedness Evaluation** (`evaluate.yml`)
   - Runs on code changes
   - Evaluates all test cases
   - Posts results to PR comments
   - Fails if pass rate is below threshold

2. **Deployment** (`main_test-streamlit-temp.yml`)
   - Only runs if groundedness evaluation passes
   - Deploys to Azure Web App

## Evaluation Metrics

### Groundedness Score (1-5)
Measures how well the generated report is supported by the source documents:
- **5**: Fully grounded - all claims supported by documents
- **4**: Mostly grounded - minor unsupported details
- **3**: Partially grounded - some unsupported claims
- **2**: Poorly grounded - significant unsupported content
- **1**: Not grounded - mostly unsupported or hallucinated

### Relevance Score (1-5)
Measures how relevant the generated content is to the assessment requirements:
- **5**: Highly relevant - directly addresses all requirements
- **4**: Mostly relevant - addresses most requirements well
- **3**: Partially relevant - addresses some requirements
- **2**: Somewhat relevant - limited relevance to requirements
- **1**: Not relevant - doesn't address requirements

### Pass Criteria

A test case passes if:
- Groundedness score ≥ threshold (default: 3.0)
- Relevance score ≥ threshold (default: 3.0)

The overall evaluation passes if:
- Pass rate ≥ threshold (default: 80%)

## Test Cases

### Case 001: ADHD Assessment
- **Age**: 12 years, Year 7
- **Documents**: School form (Form S) + Home form (Form H)
- **Focus**: Attention difficulties, hyperactivity, academic impact
- **Key Features**: Multi-setting presentation, family history

### Case 002: ASD Assessment  
- **Age**: 8 years, Year 3
- **Documents**: School form (Form S) + Autism form (Form A)
- **Focus**: Social communication, restricted interests, sensory sensitivities
- **Key Features**: Early development history, specific behavioral patterns

## Adding New Test Cases

1. Create a new directory: `test_documents/case_XXX/`
2. Add document files (`.txt`, `.pdf`, `.docx`)
3. Create `metadata.json` with case description and criteria
4. Optionally add expected output: `expected_outputs/case_XXX_expected.txt`
5. Update `evaluation_config.json` test cases list

### Metadata Format

```json
{
  "case_id": "case_003",
  "description": "Brief description of the case",
  "demographics": {
    "age": 10,
    "gender": "female",
    "school_year": "Year 5"
  },
  "presenting_concerns": [
    "Primary concern 1",
    "Primary concern 2"
  ],
  "expected_assessment_areas": [
    "Area 1 to be assessed",
    "Area 2 to be assessed"
  ],
  "key_documents": [
    "form_s.pdf",
    "other_form.pdf"
  ],
  "evaluation_criteria": {
    "must_reference_documents": true,
    "must_include_recommendations": true,
    "must_address_presenting_concerns": true
  }
}
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**: 
   - Verify app registration is created and configured
   - Check that RBAC roles are assigned correctly
   - Ensure federated credentials match your GitHub repository exactly
   
2. **Azure AI Foundry Access Denied**:
   - Confirm app registration has `Azure AI Developer` role on the AI project
   - Verify the Azure AI project and resources are in the correct subscription
   
3. **Import Errors**: Ensure Azure AI SDK is installed
4. **No Test Cases Found**: Check test document directory structure
5. **Evaluation Timeout**: Increase timeout in evaluation settings
6. **Low Scores**: Review generated reports vs. source documents

### Debugging Authentication

Test your app registration setup:

```python
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

# This should work without errors if setup is correct
credential = DefaultAzureCredential()
# Test with your AI project details
```

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check evaluation results in `evaluation_results.json` for detailed feedback.

## GitHub Secrets

Configure these secrets in your GitHub repository (**Settings** → **Secrets and variables** → **Actions**):

### For Evaluation (Required)
- `AZURE_CLIENT_ID`: App registration client ID
- `AZURE_TENANT_ID`: Azure tenant ID  
- `AZURE_SUBSCRIPTION_ID`: Azure subscription ID
- `LLM_ENDPOINT`: Azure OpenAI endpoint
- `LLM_API_KEY`: Azure OpenAI API key
- `LLM_MODEL_NAME`: Model deployment name (e.g., "gpt-4")

### For Deployment (existing)
- `AZUREAPPSERVICE_CLIENTID_*`: App service client ID
- `AZUREAPPSERVICE_TENANTID_*`: App service tenant ID
- `AZUREAPPSERVICE_SUBSCRIPTIONID_*`: App service subscription ID

**Security Note**: If using client secret instead of OIDC, also add:
- `AZURE_CLIENT_SECRET`: App registration client secret

## Best Practices

### Security
1. **Use OIDC over client secrets** for GitHub Actions authentication
2. **Principle of least privilege**: Only assign necessary RBAC roles
3. **Regular rotation**: Rotate client secrets if using them (OIDC doesn't need rotation)
4. **Monitor access**: Review Azure AD sign-in logs for the app registration

### Evaluation Quality
1. **Document Quality**: Use realistic, anonymized assessment documents
2. **Test Coverage**: Include diverse cases representing common scenarios
3. **Threshold Tuning**: Adjust thresholds based on acceptable quality levels
4. **Regular Updates**: Update test cases as assessment processes evolve
5. **Monitoring**: Track evaluation metrics over time to identify trends

## Integration with Development Workflow

1. **Feature Development**: Run evaluation locally before committing
2. **Pull Requests**: Review evaluation results in PR comments
3. **Code Reviews**: Consider evaluation impact when reviewing changes
4. **Deployment**: Only deploy when evaluation passes
5. **Monitoring**: Track production quality through regular evaluation runs