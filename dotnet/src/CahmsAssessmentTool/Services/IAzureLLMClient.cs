using CahmsAssessmentTool.Models;

namespace CahmsAssessmentTool.Services
{
    /// <summary>
    /// Interface for Azure Large Language Model client services
    /// </summary>
    public interface IAzureLLMClient
    {
        /// <summary>
        /// Gets whether the client is properly configured
        /// </summary>
        bool IsConfigured { get; }

        /// <summary>
        /// Generate assessment content based on uploaded documents
        /// </summary>
        Task<string> GenerateAssessmentAsync(DocumentRequest documentRequest);
    }
}