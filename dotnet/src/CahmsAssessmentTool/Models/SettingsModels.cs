namespace CahmsAssessmentTool.Models;

/// <summary>
/// Azure OpenAI configuration settings
/// </summary>
public class AzureOpenAISettings
{
    public const string SectionName = "AzureOpenAI";
    
    public string Endpoint { get; set; } = string.Empty;
    public string ApiKey { get; set; } = string.Empty;
    public string DeploymentName { get; set; } = string.Empty;
    public string ApiVersion { get; set; } = "2024-02-15-preview";
    public int MaxTokens { get; set; } = 4000;
    public double Temperature { get; set; } = 0.7;
    public int MaxRetries { get; set; } = 3;
    public int TimeoutSeconds { get; set; } = 120;
    
    /// <summary>
    /// Checks if all required settings are configured
    /// </summary>
    public bool IsConfigured => 
        !string.IsNullOrEmpty(Endpoint) && 
        !string.IsNullOrEmpty(ApiKey) && 
        !string.IsNullOrEmpty(DeploymentName);
}

/// <summary>
/// File upload configuration settings
/// </summary>
public class FileUploadSettings
{
    public const string SectionName = "FileUpload";
    
    public long MaxFileSize { get; set; } = 10485760; // 10 MB
    public long MaxTotalSize { get; set; } = 52428800; // 50 MB
    public string[] AllowedExtensions { get; set; } = [".pdf", ".docx", ".txt"];
    public string UploadPath { get; set; } = "uploads";
    public RequiredDocumentsSettings RequiredDocuments { get; set; } = new();
}

/// <summary>
/// Required documents configuration
/// </summary>
public class RequiredDocumentsSettings
{
    public bool FormH { get; set; }
    public bool FormS { get; set; }
    public bool FormA { get; set; }
    public bool CahmsInitial { get; set; }
    public bool NeuroDevHistory { get; set; }
    public bool FormulationDocument { get; set; }
    public bool SchoolObservation { get; set; }
    public bool SupportingInformation { get; set; }
}

/// <summary>
/// Assessment processing configuration settings
/// </summary>
public class AssessmentSettings
{
    public const string SectionName = "Assessment";
    
    public string PromptTemplatePath { get; set; } = "prompts/assessment_prompt_template.txt";
    public string SystemMessagePath { get; set; } = "prompts/system_message.txt";
    public bool EnableDetailedLogging { get; set; }
    public int MaxDocumentsPerAssessment { get; set; } = 10;
}

/// <summary>
/// General application configuration settings
/// </summary>
public class ApplicationSettings
{
    public const string SectionName = "Application";
    
    public string Name { get; set; } = "CAHMS Assessment Tool";
    public string Version { get; set; } = "1.0.0";
    public bool EnableDemoMode { get; set; }
    public int CacheTimeoutMinutes { get; set; } = 30;
}