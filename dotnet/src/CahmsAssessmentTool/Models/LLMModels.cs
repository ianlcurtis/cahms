namespace CahmsAssessmentTool.Models;

/// <summary>
/// Azure LLM Configuration
/// </summary>
public class AzureLLMConfiguration
{
    public string Endpoint { get; set; } = string.Empty;
    public string ApiKey { get; set; } = string.Empty;
    public string DeploymentName { get; set; } = string.Empty;
    public string ApiVersion { get; set; } = "2024-02-15-preview";
}

/// <summary>
/// LLM Response result
/// </summary>
public class LLMResult
{
    public string Content { get; set; } = string.Empty;
    public bool IsSuccess { get; set; }
    public string ErrorMessage { get; set; } = string.Empty;
    public LLMMetadata? Metadata { get; set; }
}

/// <summary>
/// LLM Response metadata
/// </summary>
public class LLMMetadata
{
    public int PromptTokens { get; set; }
    public int CompletionTokens { get; set; }
    public int TotalTokens { get; set; }
    public string Model { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Template execution request
/// </summary>
public class TemplateExecutionRequest
{
    public string TemplateName { get; set; } = string.Empty;
    public Dictionary<string, object> Variables { get; set; } = new();
    public int MaxTokens { get; set; } = 4000;
    public double Temperature { get; set; } = 0.3;
}
