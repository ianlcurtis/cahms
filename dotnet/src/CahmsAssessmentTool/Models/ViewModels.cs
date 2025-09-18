namespace CahmsAssessmentTool.Models;

/// <summary>
/// Configuration for individual file upload types
/// </summary>
public class FileUploadConfig
{
    public string DisplayName { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public bool IsMandatory { get; set; }
    public string Key { get; set; } = string.Empty;
}

/// <summary>
/// View model for file upload
/// </summary>
public class FileUploadViewModel
{
    // Individual file uploads for each document type
    public IFormFile? FormS { get; set; }
    public IFormFile? FormH { get; set; }
    public IFormFile? FormA { get; set; }
    public IFormFile? CahmsInitial { get; set; }
    public IFormFile? NeuroDevHistory { get; set; }
    public IFormFile? FormulationDocument { get; set; }
    public IFormFile? SchoolObservation { get; set; }
    public IFormFile? SupportingInformation { get; set; }

    // Configuration for all file types
    public Dictionary<string, FileUploadConfig> FileConfigs { get; set; } = new();

    // Parameterless constructor required for model binding
    public FileUploadViewModel()
    {
        InitializeFileConfigs(null);
    }

    // Constructor to initialize with required documents settings
    public FileUploadViewModel(RequiredDocumentsSettings? requiredDocuments)
    {
        InitializeFileConfigs(requiredDocuments);
    }

    // Method to update configurations after model binding
    public void UpdateFileConfigs(RequiredDocumentsSettings requiredDocuments)
    {
        InitializeFileConfigs(requiredDocuments);
    }

    private void InitializeFileConfigs(RequiredDocumentsSettings? requiredDocuments)
    {
        // Default all to non-mandatory if no settings provided
        var formSMandatory = requiredDocuments?.FormS ?? false;
        var formHMandatory = requiredDocuments?.FormH ?? false;
        var formAMandatory = requiredDocuments?.FormA ?? false;
        var cahmsInitialMandatory = requiredDocuments?.CahmsInitial ?? false;
        var neuroDevHistoryMandatory = requiredDocuments?.NeuroDevHistory ?? false;
        var formulationDocumentMandatory = requiredDocuments?.FormulationDocument ?? false;
        var schoolObservationMandatory = requiredDocuments?.SchoolObservation ?? false;
        var supportingInformationMandatory = requiredDocuments?.SupportingInformation ?? false;

        FileConfigs = new Dictionary<string, FileUploadConfig>
        {
            ["FormS"] = new FileUploadConfig
            {
                Key = "FormS",
                DisplayName = "Form S",
                Description = "The purpose of Form S is to gather school information for a CAMHS (Child and Adolescent Mental Health Services) Neurodevelopmental Assessment. It aims to understand the strategies and support provided to the child or young person, and why this support is necessary, to better comprehend their neurodevelopmental needs.",
                IsMandatory = formSMandatory
            },
            ["FormH"] = new FileUploadConfig
            {
                Key = "FormH",
                DisplayName = "Form H",
                Description = "The purpose of Form H is to gather school information for a CAMHS (Child and Adolescent Mental Health Services) Neurodevelopmental Assessment. It aims to understand the young person's attention, concentration, and hyperactivity levels.",
                IsMandatory = formHMandatory
            },
            ["FormA"] = new FileUploadConfig
            {
                Key = "FormA",
                DisplayName = "Form A",
                Description = "The purpose of Form A is to gather school information for a CAMHS (Child and Adolescent Mental Health Services) Neurodevelopmental Assessment. It aims to understand the young person's social interactions, communication skills, and any restricted or repetitive behaviours.",
                IsMandatory = formAMandatory
            },
            ["CahmsInitial"] = new FileUploadConfig
            {
                Key = "CahmsInitial",
                DisplayName = "CAHMS Initial Assessment Document",
                Description = "The purpose of this form is to document the initial appointment details for a young person with the CAMHS Neurodevelopmental Team. It includes information about the presenting complaint, family history, patient history, education, developmental history, and clinical observations.",
                IsMandatory = cahmsInitialMandatory
            },
            ["NeuroDevHistory"] = new FileUploadConfig
            {
                Key = "NeuroDevHistory",
                DisplayName = "Neuro Dev History",
                Description = "The purpose of this form is to gather comprehensive information about a child's developmental history, family context, and environment. It aims to understand the main challenges the child faces at school and home, the family's mental and physical health history, and the child's early years.",
                IsMandatory = neuroDevHistoryMandatory
            },
            ["FormulationDocument"] = new FileUploadConfig
            {
                Key = "FormulationDocument",
                DisplayName = "Formulation Document",
                Description = "The formulation document provides a comprehensive clinical summary and analysis of the assessment findings. It synthesizes information from all sources to develop a clear understanding of the young person's neurodevelopmental profile, strengths, challenges, and recommended interventions.",
                IsMandatory = formulationDocumentMandatory
            },
            ["SchoolObservation"] = new FileUploadConfig
            {
                Key = "SchoolObservation",
                DisplayName = "School Observation",
                Description = "This document provides additional insights from direct school observations of the child or young person in their educational environment. It can include observations of behavior, interactions, learning patterns, and social engagement.",
                IsMandatory = schoolObservationMandatory
            },
            ["SupportingInformation"] = new FileUploadConfig
            {
                Key = "SupportingInformation",
                DisplayName = "Supporting Information",
                Description = "This section allows for the upload of any additional supporting documentation that may be relevant to the neurodevelopmental assessment. This could include previous reports, specialist assessments, or other relevant clinical information.",
                IsMandatory = supportingInformationMandatory
            }
        };
    }

    // Helper method to get uploaded files as dictionary
    public Dictionary<string, IFormFile> GetUploadedFiles()
    {
        var files = new Dictionary<string, IFormFile>();
        
        if (FormS != null) files["FormS"] = FormS;
        if (FormH != null) files["FormH"] = FormH;
        if (FormA != null) files["FormA"] = FormA;
        if (CahmsInitial != null) files["CahmsInitial"] = CahmsInitial;
        if (NeuroDevHistory != null) files["NeuroDevHistory"] = NeuroDevHistory;
        if (FormulationDocument != null) files["FormulationDocument"] = FormulationDocument;
        if (SchoolObservation != null) files["SchoolObservation"] = SchoolObservation;
        if (SupportingInformation != null) files["SupportingInformation"] = SupportingInformation;
        
        return files;
    }

    // Helper method to check if all mandatory files are uploaded
    public bool AllMandatoryFilesUploaded()
    {
        var uploadedFiles = GetUploadedFiles();
        var mandatoryConfigs = FileConfigs.Where(kv => kv.Value.IsMandatory);
        
        return mandatoryConfigs.All(kv => uploadedFiles.ContainsKey(kv.Key));
    }

    // Helper method to get upload status for display
    public Dictionary<string, bool> GetUploadStatus()
    {
        var uploadedFiles = GetUploadedFiles();
        return FileConfigs.Keys.ToDictionary(key => key, key => uploadedFiles.ContainsKey(key));
    }
}

/// <summary>
/// Metadata about the assessment generation
/// </summary>
public class AssessmentMetadata
{
    public int DocumentsProcessed { get; set; }
    public int RequiredDocuments { get; set; }
    public int OptionalDocuments { get; set; }
    public string? ModelUsed { get; set; }
    public TimeSpan? GenerationTime { get; set; }
    public string? GenerationTimeFormatted { get; set; }
    public int? TotalTokens { get; set; }
    public int? PromptTokens { get; set; }
    public int? CompletionTokens { get; set; }
    public DateTime GenerationTimestamp { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// View model for assessment results
/// </summary>
public class AssessmentResultViewModel
{
    public string AssessmentContent { get; set; } = string.Empty;
    public bool IsSuccess { get; set; }
    public string ErrorMessage { get; set; } = string.Empty;
    public List<DocumentContent> ProcessedDocuments { get; set; } = new();
    public DateTime GeneratedAt { get; set; } = DateTime.UtcNow;
    public AssessmentMetadata? Metadata { get; set; }
}
