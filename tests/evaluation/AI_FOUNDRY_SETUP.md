# Fixing "AI Project client not available" Error

## Problem
You're seeing this error: `AI Project client not available. Skipping AI Foundry logging.`

## Root Cause
The Azure AI Projects SDK has changed significantly:
- **Connection string support was removed** in version 1.0.0b11+
- Your project is using version 1.0.0b12 which **requires project endpoint URLs**
- The old configuration using `AI_PROJECT_CONNECTION_STRING` no longer works

## Solution

### Option 1: Configure Project Endpoint (Recommended)

1. **Get your Azure AI Foundry project endpoint:**
   - Go to [Azure AI Foundry portal](https://ai.azure.com)
   - Select your project
   - Copy the endpoint URL from the Overview page
   - Format: `https://<your-resource>.services.ai.azure.com/api/projects/<your-project>`

2. **Set the environment variable:**
   ```bash
   export PROJECT_ENDPOINT="https://your-resource.services.ai.azure.com/api/projects/your-project"
   ```

3. **Authenticate with Azure:**
   ```bash
   az login
   ```

4. **Test the configuration:**
   ```bash
   python tests/evaluation/setup_ai_foundry.py test
   ```

### Option 2: Create .env File

1. **Copy the template:**
   ```bash
   cp .env.template .env
   ```

2. **Edit `.env` file and set:**
   ```env
   PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project
   LLM_ENDPOINT=your-azure-openai-endpoint
   LLM_API_KEY=your-api-key
   LLM_MODEL_NAME=gpt-4
   ```

3. **Load environment variables:**
   ```bash
   source .env  # or use python-dotenv in your code
   ```

### Option 3: Run Without AI Foundry Logging

If you don't need AI Foundry integration, the evaluator will work fine without it:
- Evaluation results will be saved locally as JSON files
- All evaluation functionality remains available
- You'll just see the warning message (which you can ignore)

## Verification

Run the test script to verify your setup:
```bash
python tests/evaluation/test_evaluator.py
```

You should see:
```
✓ AI Project client: Available
```

## Migration Guide

If you have old configuration, here's how to migrate:

### Old Method (Deprecated):
```env
AI_PROJECT_CONNECTION_STRING=your-connection-string
AZURE_SUBSCRIPTION_ID=your-subscription
AZURE_RESOURCE_GROUP=your-rg
AI_PROJECT_NAME=your-project
```

### New Method (Current):
```env
PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project
```

## Helper Scripts

- `tests/evaluation/setup_ai_foundry.py` - Interactive setup guide
- `tests/evaluation/test_evaluator.py` - Test your configuration
- `.env.template` - Environment variable template

## Still Having Issues?

1. **Check Azure CLI login:** `az account show`
2. **Verify endpoint format:** Must include `/api/projects/` in the URL
3. **Check permissions:** Ensure you have access to the AI Foundry project
4. **Update packages:** `pip install --upgrade azure-ai-projects azure-identity`

The evaluation will continue to work without AI Foundry logging - the warning is just informational.
