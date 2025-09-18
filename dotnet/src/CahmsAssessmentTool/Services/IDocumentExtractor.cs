using CahmsAssessmentTool.Models;

namespace CahmsAssessmentTool.Services
{
    public interface IDocumentExtractor
    {
        Task<DocumentContent> ExtractDocumentContent(IFormFile file);
        Task<List<DocumentContent>> ProcessAssessmentDocumentsAsync(FileUploadViewModel model);
        Task<List<DocumentContent>> PreprocessDocumentsAsync(List<DocumentContent> documents);
        Task<DocumentValidationResult> ValidateDocumentsAsync(List<DocumentContent> documents);
    }
}
