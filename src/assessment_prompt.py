"""
Assessment Prompt Generator for CAHMS Neurodevelopmental Assessment
This module handles the creation of prompts for LLM-based assessment report generation.
"""

import os
from typing import List, Any


# Constants for fallback messages
FALLBACK_SYSTEM_MESSAGE = "You are an expert clinician specializing in neurodevelopmental assessments for children and adolescents."
FALLBACK_PROMPT_TEMPLATE = """You are a specialist clinician working with the CAMHS (Child and Adolescent Mental Health Services) Neurodevelopmental Team.

Please analyze the provided assessment documents and generate a comprehensive neurodevelopmental assessment report.

AVAILABLE DOCUMENTS:
{document_content}

Generate the assessment report now:
"""


class AssessmentPromptGenerator:
    """Generates prompts for neurodevelopmental assessment reports using template files"""
    
    def __init__(self):
        """Initialize the prompt generator with template files"""
        self.prompt_template_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "assessment_prompt_template.txt")
        self.system_message_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "system_message.txt")
        
    def create_assessment_prompt(self, documents: List[Any]) -> str:
        """
        Create a comprehensive prompt for the LLM based on uploaded documents using template files
        
        Args:
            documents: List of processed documents
            
        Returns:
            Formatted prompt for the LLM
        """
        
        # =============================================================================
        # DOCUMENT CONTENT PREPARATION - Format documents for the template
        # =============================================================================
        
        document_content = ""
        
        for doc in documents:
            status = "REQUIRED" if doc.is_required else "OPTIONAL"
            document_content += f"\n{status} - {doc.document_type} ({doc.filename}):\n"
            document_content += f"{doc.content}\n"
            document_content += "-" * 80 + "\n"
        
        # =============================================================================
        # TEMPLATE EXECUTION - Load and prepare the prompt using the template file
        # =============================================================================
        
        try:
            # Load the prompt template file
            with open(self.prompt_template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            # Format the template with document data
            formatted_prompt = template_content.format(documents=document_content)
            
            return formatted_prompt
            
        except Exception as e:
            # Fallback to a basic prompt if template loading fails
            return FALLBACK_PROMPT_TEMPLATE.format(document_content=document_content)
    
    def create_system_message(self) -> str:
        """
        Create the system message for the LLM using the system message template file
        
        Returns:
            System message string
        """
        try:
            # Load the system message from the template file
            with open(self.system_message_path, 'r', encoding='utf-8') as f:
                system_message = f.read().strip()
            
            return system_message
                
        except Exception as e:
            # Fallback system message if template loading fails
            return FALLBACK_SYSTEM_MESSAGE
