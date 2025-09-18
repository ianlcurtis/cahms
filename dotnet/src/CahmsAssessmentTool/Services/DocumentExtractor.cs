using System.Text;
using System.Text.RegularExpressions;
using CahmsAssessmentTool.Models;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;
using PdfSharpCore.Pdf;
using PdfSharpCore.Pdf.IO;

namespace CahmsAssessmentTool.Services
{
    public class DocumentExtractor : IDocumentExtractor
    {
        private readonly ILogger<DocumentExtractor> _logger;

        public DocumentExtractor(ILogger<DocumentExtractor> logger)
        {
            _logger = logger;
        }

        public async Task<DocumentContent> ExtractDocumentContent(IFormFile file)
        {
            try
            {
                if (file == null || file.Length == 0)
                {
                    _logger.LogWarning("Empty file provided for extraction");
                    return new DocumentContent
                    {
                        Filename = file?.FileName ?? "unknown",
                        DocumentType = Path.GetExtension(file?.FileName ?? "").ToLowerInvariant(),
                        Content = $"[Empty file: {file?.FileName ?? "unknown"}]",
                        FileSize = 0,
                        UploadTimestamp = DateTime.UtcNow
                    };
                }

                _logger.LogInformation("Processing file {FileName} with size {FileSize} bytes", file.FileName, file.Length);

                using var stream = file.OpenReadStream();
                using var memoryStream = new MemoryStream();
                await stream.CopyToAsync(memoryStream);
                var fileContent = memoryStream.ToArray();

                var extension = Path.GetExtension(file.FileName).ToLowerInvariant();
                string extractedContent;

                extractedContent = extension switch
                {
                    ".txt" => await ExtractTextContentAsync(fileContent, file.FileName),
                    ".pdf" => await ExtractPdfContentAsync(fileContent, file.FileName),
                    ".doc" or ".docx" => await ExtractDocxContentAsync(fileContent, file.FileName),
                    _ => await HandleUnsupportedFormat(file.FileName)
                };

                return new DocumentContent
                {
                    Filename = file.FileName,
                    DocumentType = extension,
                    Content = extractedContent,
                    FileSize = file.Length,
                    UploadTimestamp = DateTime.UtcNow
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error extracting content from {FileName}", file.FileName);
                return new DocumentContent
                {
                    Filename = file.FileName,
                    DocumentType = Path.GetExtension(file.FileName).ToLowerInvariant(),
                    Content = $"[Error extracting content from {file.FileName}: {ex.Message}]",
                    FileSize = file.Length,
                    UploadTimestamp = DateTime.UtcNow
                };
            }
        }

        private Task<string> ExtractTextContentAsync(byte[] fileContent, string filename)
        {
            try
            {
                string text;
                try
                {
                    text = Encoding.UTF8.GetString(fileContent);
                }
                catch (Exception)
                {
                    try
                    {
                        text = Encoding.Latin1.GetString(fileContent);
                    }
                    catch (Exception ex)
                    {
                        return Task.FromResult($"[Error decoding text file {filename}: {ex.Message}]");
                    }
                }

                return Task.FromResult(CleanTextContent(text, filename));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error extracting text content from {FileName}", filename);
                return Task.FromResult($"[Error extracting text content from {filename}: {ex.Message}]");
            }
        }

        private Task<string> ExtractPdfContentAsync(byte[] fileContent, string filename)
        {
            try
            {
                if (fileContent == null || fileContent.Length == 0)
                {
                    return Task.FromResult($"[Empty file: {filename}]");
                }

                using var stream = new MemoryStream(fileContent);
                var document = PdfReader.Open(stream, PdfDocumentOpenMode.ReadOnly);

                var text = new StringBuilder();
                
                // NOTE: PdfSharpCore has limited text extraction capabilities
                // TODO: Production - Consider using a more robust PDF library like iText7 or PDFium
                for (int pageIndex = 0; pageIndex < document.PageCount; pageIndex++)
                {
                    try
                    {
                        var page = document.Pages[pageIndex];
                        // PdfSharpCore doesn't have built-in text extraction
                        // This is a placeholder - would need additional library for full PDF text extraction
                        _logger.LogWarning("PDF text extraction not fully implemented - using placeholder content for {FileName}", filename);
                        text.AppendLine($"[PDF Page {pageIndex + 1} content - full extraction requires additional library]");
                    }
                    catch (Exception pageError)
                    {
                        _logger.LogWarning(pageError, "Error extracting text from page {PageNumber} of {FileName}", pageIndex + 1, filename);
                        continue;
                    }
                }

                var extractedText = text.ToString().Trim();
                
                if (string.IsNullOrEmpty(extractedText) || extractedText.Length < 10)
                {
                    _logger.LogWarning("Very little or no text extracted from {FileName}", filename);
                    return Task.FromResult($"[PDF file {filename} appears to contain no readable text content - it may be image-based or require additional PDF processing library]");
                }

                return Task.FromResult(CleanTextContent(extractedText, filename));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error extracting PDF content from {FileName}", filename);
                return Task.FromResult($"[Error extracting PDF content from {filename}: {ex.Message}]");
            }
        }

        private Task<string> ExtractDocxContentAsync(byte[] fileContent, string filename)
        {
            try
            {
                if (fileContent == null || fileContent.Length == 0)
                {
                    return Task.FromResult($"[Empty file: {filename}]");
                }

                using var stream = new MemoryStream(fileContent);
                
                try
                {
                    using var doc = WordprocessingDocument.Open(stream, false);
                    var body = doc.MainDocumentPart?.Document?.Body;

                    if (body == null)
                    {
                        return Task.FromResult($"[DOCX file {filename} appears to have no readable content]");
                    }

                    var text = new StringBuilder();

                    // Extract text from paragraphs
                    foreach (var paragraph in body.Elements<Paragraph>())
                    {
                        var paragraphText = paragraph.InnerText.Trim();
                        if (!string.IsNullOrEmpty(paragraphText))
                        {
                            text.AppendLine(paragraphText);
                        }
                    }

                    // Extract text from tables
                    foreach (var table in body.Elements<Table>())
                    {
                        foreach (var row in table.Elements<TableRow>())
                        {
                            var rowTexts = new List<string>();
                            foreach (var cell in row.Elements<TableCell>())
                            {
                                var cellText = cell.InnerText.Trim();
                                if (!string.IsNullOrEmpty(cellText))
                                {
                                    rowTexts.Add(cellText);
                                }
                            }
                            if (rowTexts.Any())
                            {
                                text.AppendLine(string.Join("\t", rowTexts));
                            }
                        }
                    }

                    var extractedText = text.ToString().Trim();

                    if (string.IsNullOrEmpty(extractedText) || extractedText.Length < 10)
                    {
                        _logger.LogWarning("Very little or no text extracted from {FileName}", filename);
                        return Task.FromResult($"[DOCX file {filename} appears to contain no readable text content - it may be empty or corrupted]");
                    }

                    return Task.FromResult(CleanTextContent(extractedText, filename));
                }
                catch (Exception docxError)
                {
                    _logger.LogWarning(docxError, "DOCX parsing failed for {FileName}, trying text extraction", filename);

                    if (docxError.Message.ToLower().Contains("password") || docxError.Message.ToLower().Contains("encrypted"))
                    {
                        return Task.FromResult($"[DOCX file {filename} appears to be password-protected or encrypted]");
                    }

                    // Fallback to plain text extraction
                    foreach (var encoding in new[] { Encoding.UTF8, Encoding.Latin1 })
                    {
                        try
                        {
                            var textContent = encoding.GetString(fileContent).Trim();
                            if (textContent.Length > 50)
                            {
                                return Task.FromResult(CleanTextContent(textContent, filename));
                            }
                        }
                        catch
                        {
                            continue;
                        }
                    }

                    return Task.FromResult($"[Could not extract content from {filename} - file may be corrupted or in unsupported format]");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error extracting DOCX content from {FileName}", filename);
                return Task.FromResult($"[Error extracting DOCX content from {filename}: {ex.Message}]");
            }
        }

        private Task<string> HandleUnsupportedFormat(string filename)
        {
            _logger.LogWarning("Unsupported file format: {FileName}", filename);
            throw new NotImplementedException($"Document extraction for file type '{Path.GetExtension(filename)}' is not implemented. Supported formats: .txt, .pdf, .doc, .docx");
        }

        private string CleanTextContent(string text, string filename)
        {
            // Remove null bytes and problematic characters
            text = text.Replace("\0", "");

            // Remove excessive whitespace
            text = Regex.Replace(text, @"\s+", " ");
            text = Regex.Replace(text, @"\n\s*\n", "\n\n");
            text = text.Trim();

            // Validate content
            if (string.IsNullOrEmpty(text))
            {
                return $"[File {filename} appears to be empty after processing]";
            }

            if (text.Length < 10)
            {
                return $"[File {filename} contains very little readable content: {text}]";
            }

            // Check for binary data patterns
            var nonPrintableCount = text.Count(c => char.IsControl(c) && c != '\n' && c != '\r' && c != '\t');
            if (text.Length > 0 && (double)nonPrintableCount / text.Length > 0.5)
            {
                return $"[File {filename} appears to contain binary data or is corrupted]";
            }

            return text;
        }

        public async Task<List<DocumentContent>> ProcessAssessmentDocumentsAsync(FileUploadViewModel model)
        {
            var documents = new List<DocumentContent>();

            var uploadedFiles = model.GetUploadedFiles();
            if (!uploadedFiles.Any())
            {
                return documents;
            }

            foreach (var kvp in uploadedFiles)
            {
                var file = kvp.Value;
                var fileKey = kvp.Key;
                var config = model.FileConfigs.ContainsKey(fileKey) ? model.FileConfigs[fileKey] : null;
                
                try
                {
                    var documentContent = await ExtractDocumentContent(file);
                    
                    // Update document type if we have config info
                    if (config != null)
                    {
                        documentContent.DocumentType = config.DisplayName;
                        documentContent.IsRequired = config.IsMandatory;
                    }
                    
                    documents.Add(documentContent);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing file: {FileName}", file.FileName);
                    // TODO: Production - Add proper error handling and user feedback
                    
                    // Add error document to maintain file tracking
                    documents.Add(new DocumentContent
                    {
                        Filename = file.FileName,
                        DocumentType = config?.DisplayName ?? Path.GetExtension(file.FileName).ToLowerInvariant(),
                        Content = $"[Error processing {file.FileName}: {ex.Message}]",
                        FileSize = file.Length,
                        UploadTimestamp = DateTime.UtcNow
                    });
                }
            }

            return documents;
        }

        public Task<List<DocumentContent>> PreprocessDocumentsAsync(List<DocumentContent> documents)
        {
            _logger.LogInformation("Preprocessing {DocumentCount} documents", documents.Count);
            var processedDocuments = new List<DocumentContent>();

            for (int i = 0; i < documents.Count; i++)
            {
                var doc = documents[i];
                _logger.LogDebug("Processing document {DocumentNumber}: {Filename}", i + 1, doc.Filename);

                try
                {
                    // Check if document has processing errors
                    if (doc.Content.StartsWith('[') && doc.Content.EndsWith(']'))
                    {
                        _logger.LogWarning("Document {Filename} has processing error: {Content}", doc.Filename, doc.Content);
                    }
                    else
                    {
                        // Check for suspiciously short content
                        if (doc.Content.Trim().Length < 50)
                        {
                            _logger.LogWarning("Document {Filename} has very short content: {Length} characters", doc.Filename, doc.Content.Length);
                        }

                        // Check for binary data patterns
                        var nonPrintableCount = doc.Content.Count(c => c < 32 && c != '\n' && c != '\r' && c != '\t');
                        var nonPrintableRatio = doc.Content.Length > 0 ? (double)nonPrintableCount / doc.Content.Length : 0;
                        
                        if (nonPrintableRatio > 0.3)
                        {
                            _logger.LogWarning("Document {Filename} contains high ratio of non-printable characters: {Ratio:P}", doc.Filename, nonPrintableRatio);
                        }

                        _logger.LogDebug("Document {DocumentNumber} processed successfully ({Length} chars)", i + 1, doc.Content.Length);
                    }

                    processedDocuments.Add(doc);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing document {Filename}", doc.Filename);
                    processedDocuments.Add(doc);
                }
            }

            _logger.LogInformation("Preprocessing complete, {ProcessedCount} documents ready", processedDocuments.Count);
            return Task.FromResult(processedDocuments);
        }

        public Task<DocumentValidationResult> ValidateDocumentsAsync(List<DocumentContent> documents)
        {
            var validationResult = new DocumentValidationResult
            {
                IsValid = true,
                Warnings = new List<string>(),
                Errors = new List<string>(),
                DocumentAnalysis = new Dictionary<string, DocumentAnalysis>()
            };

            // Check for required documents that are missing content
            var requiredDocs = documents.Where(doc => doc.IsRequired).ToList();
            var missingRequired = requiredDocs
                .Where(doc => string.IsNullOrWhiteSpace(doc.Content))
                .Select(doc => doc.DocumentType)
                .ToList();

            if (missingRequired.Any())
            {
                validationResult.IsValid = false;
                validationResult.Errors.Add($"Required documents have no content: {string.Join(", ", missingRequired)}");
            }

            // Analyze each document
            foreach (var doc in documents)
            {
                var wordCount = doc.Content?.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries).Length ?? 0;
                var hasContent = !string.IsNullOrWhiteSpace(doc.Content);
                var fileSizeKb = doc.Content?.Length ?? 0;

                var analysis = new DocumentAnalysis
                {
                    WordCount = wordCount,
                    HasContent = hasContent,
                    FileSizeKb = fileSizeKb / 1024.0
                };

                if (!analysis.HasContent)
                {
                    if (doc.IsRequired)
                    {
                        validationResult.Errors.Add($"{doc.DocumentType} is required but appears to be empty");
                        validationResult.IsValid = false;
                    }
                    else
                    {
                        validationResult.Warnings.Add($"{doc.DocumentType} appears to be empty");
                    }
                }
                else if (analysis.WordCount < 10)
                {
                    validationResult.Warnings.Add($"{doc.DocumentType} has very little content ({analysis.WordCount} words)");
                }

                validationResult.DocumentAnalysis[doc.DocumentType] = analysis;
            }

            return Task.FromResult(validationResult);
        }
    }
}
