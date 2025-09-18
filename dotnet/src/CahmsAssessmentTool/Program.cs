using CahmsAssessmentTool.Models;
using CahmsAssessmentTool.Services;

// Load environment variables from .env file
DotNetEnv.Env.Load("../../.env");

var builder = WebApplication.CreateBuilder(args);

// Add environment variables to configuration
builder.Configuration.AddEnvironmentVariables();

// Add services to the container.
builder.Services.AddControllersWithViews();

// Configure settings models from appsettings.json and environment variables
builder.Services.Configure<AzureOpenAISettings>(options =>
{
    builder.Configuration.GetSection(AzureOpenAISettings.SectionName).Bind(options);
    
    // Override with environment variables if they exist
    if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")))
        options.Endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!;
    if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY")))
        options.ApiKey = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY")!;
    if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME")))
        options.DeploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME")!;
    if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("AZURE_OPENAI_API_VERSION")))
        options.ApiVersion = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_VERSION")!;
    if (int.TryParse(Environment.GetEnvironmentVariable("AZURE_OPENAI_MAX_TOKENS"), out var maxTokens))
        options.MaxTokens = maxTokens;
    if (double.TryParse(Environment.GetEnvironmentVariable("AZURE_OPENAI_TEMPERATURE"), out var temperature))
        options.Temperature = temperature;
    if (int.TryParse(Environment.GetEnvironmentVariable("AZURE_OPENAI_MAX_RETRIES"), out var maxRetries))
        options.MaxRetries = maxRetries;
    if (int.TryParse(Environment.GetEnvironmentVariable("AZURE_OPENAI_TIMEOUT_SECONDS"), out var timeoutSeconds))
        options.TimeoutSeconds = timeoutSeconds;
});
builder.Services.Configure<FileUploadSettings>(options =>
{
    builder.Configuration.GetSection(FileUploadSettings.SectionName).Bind(options);
    
    // Override with environment variables if they exist
    if (long.TryParse(Environment.GetEnvironmentVariable("FILE_UPLOAD_MAX_FILE_SIZE"), out var maxFileSize))
        options.MaxFileSize = maxFileSize;
    if (long.TryParse(Environment.GetEnvironmentVariable("FILE_UPLOAD_MAX_TOTAL_SIZE"), out var maxTotalSize))
        options.MaxTotalSize = maxTotalSize;
    if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("FILE_UPLOAD_ALLOWED_EXTENSIONS")))
        options.AllowedExtensions = Environment.GetEnvironmentVariable("FILE_UPLOAD_ALLOWED_EXTENSIONS")!.Split(',');
    if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("FILE_UPLOAD_PATH")))
        options.UploadPath = Environment.GetEnvironmentVariable("FILE_UPLOAD_PATH")!;
    
    // Required documents overrides
    if (bool.TryParse(Environment.GetEnvironmentVariable("MANDATORY_FORM_H"), out var formH))
        options.RequiredDocuments.FormH = formH;
    if (bool.TryParse(Environment.GetEnvironmentVariable("MANDATORY_FORM_S"), out var formS))
        options.RequiredDocuments.FormS = formS;
    if (bool.TryParse(Environment.GetEnvironmentVariable("MANDATORY_FORM_A"), out var formA))
        options.RequiredDocuments.FormA = formA;
    if (bool.TryParse(Environment.GetEnvironmentVariable("MANDATORY_CAHMS_INITIAL"), out var cahmsInitial))
        options.RequiredDocuments.CahmsInitial = cahmsInitial;
    if (bool.TryParse(Environment.GetEnvironmentVariable("MANDATORY_NEURO_DEV_HISTORY"), out var neuroDevHistory))
        options.RequiredDocuments.NeuroDevHistory = neuroDevHistory;
    if (bool.TryParse(Environment.GetEnvironmentVariable("MANDATORY_FORMULATION_DOCUMENT"), out var formulationDocument))
        options.RequiredDocuments.FormulationDocument = formulationDocument;
    if (bool.TryParse(Environment.GetEnvironmentVariable("MANDATORY_SCHOOL_OBSERVATION"), out var schoolObservation))
        options.RequiredDocuments.SchoolObservation = schoolObservation;
    if (bool.TryParse(Environment.GetEnvironmentVariable("MANDATORY_SUPPORTING_INFORMATION"), out var supportingInformation))
        options.RequiredDocuments.SupportingInformation = supportingInformation;
});
builder.Services.Configure<AssessmentSettings>(options =>
{
    builder.Configuration.GetSection(AssessmentSettings.SectionName).Bind(options);
    
    // Override with environment variables if they exist
    if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ASSESSMENT_PROMPT_TEMPLATE_PATH")))
        options.PromptTemplatePath = Environment.GetEnvironmentVariable("ASSESSMENT_PROMPT_TEMPLATE_PATH")!;
    if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ASSESSMENT_SYSTEM_MESSAGE_PATH")))
        options.SystemMessagePath = Environment.GetEnvironmentVariable("ASSESSMENT_SYSTEM_MESSAGE_PATH")!;
    if (bool.TryParse(Environment.GetEnvironmentVariable("ASSESSMENT_ENABLE_DETAILED_LOGGING"), out var logging))
        options.EnableDetailedLogging = logging;
    if (int.TryParse(Environment.GetEnvironmentVariable("ASSESSMENT_MAX_DOCUMENTS"), out var maxDocs))
        options.MaxDocumentsPerAssessment = maxDocs;
});
builder.Services.Configure<ApplicationSettings>(options =>
{
    builder.Configuration.GetSection(ApplicationSettings.SectionName).Bind(options);
    
    // Override with environment variables if they exist
    if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("APPLICATION_NAME")))
        options.Name = Environment.GetEnvironmentVariable("APPLICATION_NAME")!;
    if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("APPLICATION_VERSION")))
        options.Version = Environment.GetEnvironmentVariable("APPLICATION_VERSION")!;
    if (bool.TryParse(Environment.GetEnvironmentVariable("APPLICATION_ENABLE_DEMO_MODE"), out var demo))
        options.EnableDemoMode = demo;
    if (int.TryParse(Environment.GetEnvironmentVariable("APPLICATION_CACHE_TIMEOUT_MINUTES"), out var timeout))
        options.CacheTimeoutMinutes = timeout;
});

// Configure Azure LLM settings (legacy compatibility)
// Remove old configuration - now using strongly-typed settings from appsettings.json
// builder.Services.Configure<AzureLLMConfiguration>(options =>
// {
//     options.Endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT") ?? "";
//     options.ApiKey = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY") ?? "";
//     options.DeploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") ?? "";
//     options.ApiVersion = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_VERSION") ?? "2024-02-15-preview";
// });

// Register services
builder.Services.AddScoped<IDocumentExtractor, DocumentExtractor>();
builder.Services.AddScoped<IAssessmentPromptGenerator, AssessmentPromptGenerator>();
builder.Services.AddScoped<IAzureLLMClient, AzureLLMClientSemanticKernel>();

// Add logging
builder.Services.AddLogging();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();

// Add request logging middleware
app.Use(async (context, next) =>
{
    var logger = context.RequestServices.GetRequiredService<ILogger<Program>>();
    logger.LogInformation("=== INCOMING REQUEST ===");
    logger.LogInformation("Method: {Method}", context.Request.Method);
    logger.LogInformation("Path: {Path}", context.Request.Path);
    logger.LogInformation("Query: {Query}", context.Request.QueryString);
    logger.LogInformation("Content-Type: {ContentType}", context.Request.ContentType);
    logger.LogInformation("Has Form: {HasForm}", context.Request.HasFormContentType);
    logger.LogInformation("Content-Length: {ContentLength}", context.Request.ContentLength);
    
    await next();
    
    logger.LogInformation("Response Status: {StatusCode}", context.Response.StatusCode);
    logger.LogInformation("=== REQUEST COMPLETED ===");
});

app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();
