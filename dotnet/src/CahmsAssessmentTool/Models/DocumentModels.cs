namespace CahmsAssessmentTool.Models;

/// <summary>
/// Represents extracted document content
/// </summary>
public class DocumentContent
{
    public string DocumentType { get; set; } = string.Empty;
    public string Content { get; set; } = string.Empty;
    public string Filename { get; set; } = string.Empty;
    public long FileSize { get; set; }
    public bool IsRequired { get; set; } = false;
    public DateTime UploadTimestamp { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Configuration for file handling
/// </summary>
public class FileConfiguration
{
    public bool Required { get; set; }
    public string Description { get; set; } = string.Empty;
    public string[] AllowedExtensions { get; set; } = Array.Empty<string>();
}

/// <summary>
/// Configuration for mandatory uploads
/// </summary>
public class MandatoryUploadsConfiguration
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
/// Document processing request
/// </summary>
public class DocumentRequest
{
    public List<DocumentContent> Documents { get; set; } = new();
    public string AssessmentType { get; set; } = string.Empty;
}

/// <summary>
/// Document validation results
/// </summary>
public class DocumentValidationResult
{
    public bool IsValid { get; set; } = true;
    public List<string> Warnings { get; set; } = new();
    public List<string> Errors { get; set; } = new();
    public Dictionary<string, DocumentAnalysis> DocumentAnalysis { get; set; } = new();
}

/// <summary>
/// Analysis of individual document
/// </summary>
public class DocumentAnalysis
{
    public int WordCount { get; set; }
    public bool HasContent { get; set; }
    public double FileSizeKb { get; set; }
}
