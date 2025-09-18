using CahmsAssessmentTool.Models;
using Microsoft.Extensions.Options;
using System.Text;

namespace CahmsAssessmentTool.Services
{
    /// <summary>
    /// Assessment Prompt Generator - port of assessment_prompt.py
    /// Generates structured prompts for CAHMS assessments based on document content
    /// </summary>
    public class AssessmentPromptGenerator : IAssessmentPromptGenerator
    {
        private readonly ILogger<AssessmentPromptGenerator> _logger;
        private readonly AssessmentSettings _assessmentSettings;
        private readonly string _systemMessagePath;
        private readonly string _assessmentTemplatePath;

        public AssessmentPromptGenerator(
            ILogger<AssessmentPromptGenerator> logger,
            IOptions<AssessmentSettings> assessmentSettings)
        {
            _logger = logger;
            _assessmentSettings = assessmentSettings.Value;
            
            // Find the workspace root by looking for the .sln file
            string currentDirectory = Directory.GetCurrentDirectory();
            string workspaceRoot = FindWorkspaceRoot(currentDirectory);
            
            // Use configured paths from settings
            _systemMessagePath = Path.IsPathRooted(_assessmentSettings.SystemMessagePath) 
                ? _assessmentSettings.SystemMessagePath 
                : Path.Combine(workspaceRoot, _assessmentSettings.SystemMessagePath);
                
            _assessmentTemplatePath = Path.IsPathRooted(_assessmentSettings.PromptTemplatePath) 
                ? _assessmentSettings.PromptTemplatePath 
                : Path.Combine(workspaceRoot, _assessmentSettings.PromptTemplatePath);
        }

        /// <summary>
        /// Find the workspace root by looking for the .sln file
        /// </summary>
        private string FindWorkspaceRoot(string startPath)
        {
            var directory = new DirectoryInfo(startPath);
            while (directory != null)
            {
                if (directory.GetFiles("*.sln").Any())
                {
                    return directory.FullName;
                }
                directory = directory.Parent;
            }
            
            // Fallback: assume we're in the project directory and go up to find workspace root
            return Path.GetFullPath(Path.Combine(startPath, "..", "..", "..", ".."));
        }

        /// <summary>
        /// Generate a structured prompt for assessment based on document content
        /// </summary>
        public async Task<string> GenerateAssessmentPromptAsync(List<DocumentContent> documents, string? assessmentType = null)
        {
            try
            {
                // Load the assessment template
                var template = await LoadAssessmentTemplateAsync();
                
                // Build document content section in the format expected by the template
                var documentSection = BuildDocumentSection(documents);
                
                // Replace the {documents} placeholder in the template (Python format)
                var prompt = template.Replace("{documents}", documentSection);

                return prompt;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to generate assessment prompt");
                throw new InvalidOperationException($"Failed to generate assessment prompt: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Get the system message for the assessment
        /// </summary>
        public string GetSystemMessage()
        {
            try
            {
                if (File.Exists(_systemMessagePath))
                {
                    return File.ReadAllText(_systemMessagePath);
                }
                else
                {
                    var error = $"System message file not found at: {_systemMessagePath}";
                    _logger.LogError(error);
                    throw new FileNotFoundException(error);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load system message from file: {Path}", _systemMessagePath);
                throw;
            }
        }

        /// <summary>
        /// Load assessment template from file
        /// </summary>
        private async Task<string> LoadAssessmentTemplateAsync()
        {
            try
            {
                if (File.Exists(_assessmentTemplatePath))
                {
                    return await File.ReadAllTextAsync(_assessmentTemplatePath);
                }
                else
                {
                    var error = $"Assessment template file not found at: {_assessmentTemplatePath}";
                    _logger.LogError(error);
                    throw new FileNotFoundException(error);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load assessment template from file: {Path}", _assessmentTemplatePath);
                throw;
            }
        }

        /// <summary>
        /// Build the document content section for the prompt in Python format
        /// </summary>
        private string BuildDocumentSection(List<DocumentContent> documents)
        {
            if (documents == null || !documents.Any())
            {
                return "No documents provided.";
            }

            var section = new StringBuilder();

            foreach (var doc in documents.Where(d => !string.IsNullOrWhiteSpace(d.Content)))
            {
                // Format similar to Python: "REQUIRED/OPTIONAL - DocumentType (filename):"
                string status = "REQUIRED"; // TODO: Add IsRequired property to DocumentContent if needed
                section.AppendLine($"{status} - {doc.DocumentType} ({doc.Filename}):");
                section.AppendLine(doc.Content);
                section.AppendLine(new string('-', 80));
            }

            return section.ToString();
        }

    }
}