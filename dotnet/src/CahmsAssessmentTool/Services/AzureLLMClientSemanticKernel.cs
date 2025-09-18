using CahmsAssessmentTool.Models;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.Extensions.Options;

namespace CahmsAssessmentTool.Services
{
    /// <summary>
    /// Semantic Kernel-based Azure LLM client - port of azure_llm_client_sk.py
    /// Handles communication with Azure OpenAI using Microsoft Semantic Kernel
    /// </summary>
    public class AzureLLMClientSemanticKernel : IAzureLLMClient
    {
        private readonly AzureOpenAISettings _config;
        private readonly ILogger<AzureLLMClientSemanticKernel> _logger;
        private readonly Kernel _kernel;
        private readonly IChatCompletionService? _chatService;
        private readonly IDocumentExtractor? _documentExtractor;
        private readonly IAssessmentPromptGenerator? _promptGenerator;

        public bool IsConfigured { get; private set; }

        public AzureLLMClientSemanticKernel(
            IOptions<AzureOpenAISettings> config,
            ILogger<AzureLLMClientSemanticKernel> logger,
            IDocumentExtractor? documentExtractor = null,
            IAssessmentPromptGenerator? promptGenerator = null)
        {
            _config = config.Value;
            _logger = logger;
            _documentExtractor = documentExtractor;
            _promptGenerator = promptGenerator;

            // Initialize Semantic Kernel
            _kernel = InitializeKernel();
            
            // Only get chat service if kernel is properly configured
            _chatService = _kernel.Services.GetService<IChatCompletionService>();
            
            IsConfigured = _chatService is not null && !string.IsNullOrEmpty(_config.Endpoint) && !string.IsNullOrEmpty(_config.ApiKey);
        }

        /// <summary>
        /// Initialize the Semantic Kernel with Azure OpenAI configuration
        /// </summary>
        private Kernel InitializeKernel()
        {
            try
            {
                // Validate basic configuration first
                if (string.IsNullOrEmpty(_config.Endpoint) || 
                    string.IsNullOrEmpty(_config.ApiKey) ||
                    string.IsNullOrEmpty(_config.DeploymentName))
                {
                    _logger.LogWarning("Azure OpenAI credentials not configured - some settings are missing");
                    return CreateEmptyKernel();
                }

                // Validate endpoint URL format
                if (!Uri.TryCreate(_config.Endpoint, UriKind.Absolute, out var endpointUri) ||
                    (!endpointUri.Scheme.Equals("http", StringComparison.OrdinalIgnoreCase) &&
                     !endpointUri.Scheme.Equals("https", StringComparison.OrdinalIgnoreCase)))
                {
                    _logger.LogWarning("Azure OpenAI endpoint is not a valid URL: {Endpoint}", _config.Endpoint);
                    return CreateEmptyKernel();
                }

                // Check for placeholder values
                if (_config.Endpoint.Contains("your-dev-openai") || 
                    _config.ApiKey.Contains("placeholder") ||
                    _config.ApiKey.Contains("dev-api-key"))
                {
                    _logger.LogWarning("Azure OpenAI using placeholder configuration - service will not be functional");
                    return CreateEmptyKernel();
                }

                var builder = Kernel.CreateBuilder();
                
                builder.AddAzureOpenAIChatCompletion(
                    deploymentName: _config.DeploymentName,
                    endpoint: _config.Endpoint,
                    apiKey: _config.ApiKey,
                    apiVersion: _config.ApiVersion);

                _logger.LogInformation("Azure OpenAI Semantic Kernel initialized successfully");
                return builder.Build();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize Semantic Kernel - using empty kernel");
                return CreateEmptyKernel();
            }
        }

        /// <summary>
        /// Create an empty kernel when Azure OpenAI is not properly configured
        /// </summary>
        private Kernel CreateEmptyKernel()
        {
            return Kernel.CreateBuilder().Build();
        }

        /// <summary>
        /// Generate assessment content based on uploaded documents
        /// </summary>
        public async Task<string> GenerateAssessmentAsync(DocumentRequest documentRequest)
        {
            try
            {
                _logger.LogInformation("Starting assessment generation...");
                
                if (!IsConfigured || _chatService is null)
                {
                    throw new InvalidOperationException("Azure LLM client is not properly configured");
                }

                if (_promptGenerator == null)
                {
                    throw new InvalidOperationException("Assessment prompt generator is not available");
                }

                _logger.LogInformation("Generating assessment prompt...");
                // Generate the assessment prompt
                var prompt = await _promptGenerator.GenerateAssessmentPromptAsync(
                    documentRequest.Documents, 
                    documentRequest.AssessmentType);
                
                _logger.LogDebug("Generated prompt length: {PromptLength} characters", prompt.Length);

                // Get system message
                var systemMessage = _promptGenerator.GetSystemMessage();
                _logger.LogDebug("System message length: {SystemMessageLength} characters", systemMessage.Length);

                // Create chat history
                var chatHistory = new ChatHistory();
                chatHistory.AddSystemMessage(systemMessage);
                chatHistory.AddUserMessage(prompt);

                // Configure execution settings with timeout
                var executionSettings = new OpenAIPromptExecutionSettings
                {
                    MaxTokens = 4000,
                    Temperature = 0.3,
                    TopP = 1.0,
                    FrequencyPenalty = 0.0,
                    PresencePenalty = 0.0
                };

                _logger.LogInformation("Sending request to Azure OpenAI API...");
                
                // Create cancellation token with shorter timeout for testing
                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(60));
                
                // Generate response
                var response = await _chatService.GetChatMessageContentAsync(
                    chatHistory,
                    executionSettings,
                    cancellationToken: cts.Token);

                _logger.LogInformation("Received response from Azure OpenAI API. Length: {ResponseLength} characters", 
                    response.Content?.Length ?? 0);

                return response.Content ?? string.Empty;
            }
            catch (OperationCanceledException ex)
            {
                _logger.LogError(ex, "Assessment generation timed out after 120 seconds");
                throw new TimeoutException("The assessment generation request timed out. Please try again or contact support.", ex);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to generate assessment. Exception type: {ExceptionType}, Message: {Message}", 
                    ex.GetType().Name, ex.Message);
                throw new InvalidOperationException($"Failed to generate assessment: {ex.Message}", ex);
            }
        }
    }
}