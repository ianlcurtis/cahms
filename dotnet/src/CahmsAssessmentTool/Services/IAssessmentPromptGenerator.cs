using CahmsAssessmentTool.Models;

namespace CahmsAssessmentTool.Services
{
    /// <summary>
    /// Interface for generating assessment prompts
    /// </summary>
    public interface IAssessmentPromptGenerator
    {
        /// <summary>
        /// Generate a structured prompt for assessment based on document content
        /// </summary>
        Task<string> GenerateAssessmentPromptAsync(List<DocumentContent> documents, string? assessmentType = null);

        /// <summary>
        /// Get the system message for the assessment
        /// </summary>
        string GetSystemMessage();
    }
}