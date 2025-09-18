using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using CahmsAssessmentTool.Models;
using CahmsAssessmentTool.Services;

namespace CahmsAssessmentTool.Controllers;

public class HomeController : Controller
{
    private readonly ILogger<HomeController> _logger;
    private readonly IDocumentExtractor _documentExtractor;
    private readonly IAzureLLMClient _llmClient;
    private readonly AzureOpenAISettings _azureOpenAISettings;
    private readonly FileUploadSettings _fileUploadSettings;
    private readonly AssessmentSettings _assessmentSettings;
    private readonly ApplicationSettings _applicationSettings;

    public HomeController(
        ILogger<HomeController> logger,
        IDocumentExtractor documentExtractor,
        IAzureLLMClient llmClient,
        IOptions<AzureOpenAISettings> azureOpenAISettings,
        IOptions<FileUploadSettings> fileUploadSettings,
        IOptions<AssessmentSettings> assessmentSettings,
        IOptions<ApplicationSettings> applicationSettings)
    {
        _logger = logger;
        _documentExtractor = documentExtractor;
        _llmClient = llmClient;
        _azureOpenAISettings = azureOpenAISettings.Value;
        _fileUploadSettings = fileUploadSettings.Value;
        _assessmentSettings = assessmentSettings.Value;
        _applicationSettings = applicationSettings.Value;
    }

    public IActionResult Index()
    {
       var viewModel = new FileUploadViewModel(_fileUploadSettings.RequiredDocuments);
       return View(viewModel);
    }

    [HttpPost]
    [Route("Home/ProcessAssessment")]
    public async Task<IActionResult> ProcessAssessment(FileUploadViewModel model)
    {
        // Ensure model has the correct settings initialized since model binding doesn't preserve FileConfigs
        model.UpdateFileConfigs(_fileUploadSettings.RequiredDocuments);
        // Log all ModelState errors if validation fails
        if (!ModelState.IsValid)
        {
            _logger.LogWarning("ModelState validation failed:");
            foreach (var modelError in ModelState)
            {
                _logger.LogWarning("Key: {Key}, Errors: {Errors}", 
                    modelError.Key, 
                    string.Join(", ", modelError.Value.Errors.Select(e => e.ErrorMessage)));
            }
            
            // Re-initialize FileConfigs with settings since model binding doesn't preserve it
            model.UpdateFileConfigs(_fileUploadSettings.RequiredDocuments);
            
            // Return to the same view with validation errors
            return View("Index", model);
        }
        
        
        var uploadedFiles = model.GetUploadedFiles();
        _logger.LogInformation("Uploaded files: {Count}", uploadedFiles.Count);
        
        var result = new AssessmentResultViewModel();
        var startTime = DateTime.UtcNow;

        try
        {
            _logger.LogInformation("Starting assessment processing...");
            
            if (!uploadedFiles.Any())
            {
                _logger.LogWarning("No files were uploaded");
                result.IsSuccess = false;
                result.ErrorMessage = "No files were uploaded.";
                return View("Result", result);
            }

            // Check if all mandatory files are uploaded
            if (!model.AllMandatoryFilesUploaded())
            {
                _logger.LogWarning("Not all mandatory files were uploaded");
                result.IsSuccess = false;
                result.ErrorMessage = "Please upload all required documents before generating the report.";
                return View("Result", result);
            }

            _logger.LogInformation("Processing {FileCount} uploaded files", uploadedFiles.Count);

            // Check if LLM client is configured
            if (!_llmClient.IsConfigured)
            {
                _logger.LogError("Azure OpenAI client is not configured");
                result.IsSuccess = false;
                result.ErrorMessage = "Azure OpenAI is not configured. Please check your environment variables.";
                return View("Result", result);
            }

            _logger.LogInformation("Azure OpenAI client is configured, proceeding with document extraction");

            // Process uploaded documents using the new structure
            var documents = await ProcessUploadedFilesAsync(uploadedFiles, model.FileConfigs);
            result.ProcessedDocuments = documents;

            if (!documents.Any())
            {
                _logger.LogWarning("No content could be extracted from uploaded files");
                result.IsSuccess = false;
                result.ErrorMessage = "No content could be extracted from the uploaded files.";
                return View("Result", result);
            }

            _logger.LogInformation("Extracted content from {DocumentCount} documents", documents.Count);

            // Create document request
            var documentRequest = new DocumentRequest
            {
                Documents = documents,
                AssessmentType = "Comprehensive CAHMS Assessment"
            };

            
            // Generate assessment
            var assessmentContent = await _llmClient.GenerateAssessmentAsync(documentRequest);
            var endTime = DateTime.UtcNow;
            var duration = endTime - startTime;
            
            _logger.LogInformation("Assessment generation completed successfully");
            
            // Create metadata
            var requiredDocs = model.FileConfigs.Count(kv => kv.Value.IsMandatory);
            var optionalDocs = uploadedFiles.Count - requiredDocs;
            
            result.Metadata = new AssessmentMetadata
            {
                DocumentsProcessed = documents.Count,
                RequiredDocuments = requiredDocs,
                OptionalDocuments = optionalDocs,
                ModelUsed = _azureOpenAISettings.DeploymentName,
                GenerationTime = duration,
                GenerationTimeFormatted = $"{(int)duration.TotalMinutes}m {duration.Seconds}s",
                GenerationTimestamp = endTime,
                // TODO: Add token usage if available from the LLM client
                TotalTokens = null,
                PromptTokens = null,
                CompletionTokens = null
            };
            
            result.IsSuccess = true;
            result.AssessmentContent = assessmentContent;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing assessment");
            result.IsSuccess = false;
            result.ErrorMessage = $"An error occurred while processing the assessment: {ex.Message}";
        }

        _logger.LogInformation("Returning result view");
        return View("Result", result);
    }

    /// <summary>
    /// Process uploaded files with the new file configuration structure
    /// </summary>
    private async Task<List<DocumentContent>> ProcessUploadedFilesAsync(
        Dictionary<string, IFormFile> uploadedFiles, 
        Dictionary<string, FileUploadConfig> fileConfigs)
    {
        var documents = new List<DocumentContent>();
        
        foreach (var kvp in uploadedFiles)
        {
            var fileKey = kvp.Key;
            var file = kvp.Value;
            var config = fileConfigs.ContainsKey(fileKey) ? fileConfigs[fileKey] : null;
            
            if (config == null) continue;
            
            try
            {
                // Use the existing document extractor to process the file
                var documentContent = await _documentExtractor.ExtractDocumentContent(file);
                
                if (!string.IsNullOrWhiteSpace(documentContent.Content))
                {
                    // Update the document with our configuration info
                    documentContent.DocumentType = config.DisplayName;
                    documentContent.IsRequired = config.IsMandatory;
                    
                    documents.Add(documentContent);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to process file {FileName} for {DocumentType}", 
                    file.FileName, config.DisplayName);
            }
        }
        
        return documents;
    }
}
