"""
Assessment Prompt Generator for CAHMS Neurodevelopmental Assessment
This module handles the creation of prompts for LLM-based assessment report generation.
"""

import os
from typing import List, Any
from prompty import load, prepare


# Constants for fallback messages
FALLBACK_SYSTEM_MESSAGE = "You are an expert clinician specializing in neurodevelopmental assessments for children and adolescents."
FALLBACK_PROMPT_TEMPLATE = """You are a specialist clinician working with the CAMHS (Child and Adolescent Mental Health Services) Neurodevelopmental Team.

Please analyze the provided assessment documents and generate a comprehensive neurodevelopmental assessment report.

AVAILABLE DOCUMENTS:
{document_content}

Generate the assessment report now:
"""


class AssessmentPromptGenerator:
    """Generates prompts for neurodevelopmental assessment reports using Prompty"""
    
    def __init__(self):
        """Initialize the prompt generator with the prompty file"""
        self.prompty_file_path = os.path.join(os.path.dirname(__file__), "assessment_report.prompty")
        
    def create_assessment_prompt(self, documents: List[Any]) -> str:
        """
        Create a comprehensive prompt for the LLM based on uploaded documents using Prompty
        
        Args:
            documents: List of processed documents
            
        Returns:
            Formatted prompt for the LLM
        """
        
        # =============================================================================
        # DOCUMENT CONTENT PREPARATION - Format documents for the prompty template
        # =============================================================================
        
        document_content = ""
        
        for doc in documents:
            status = "REQUIRED" if doc.is_required else "OPTIONAL"
            document_content += f"\n{status} - {doc.document_type} ({doc.filename}):\n"
            document_content += f"{doc.content}\n"
            document_content += "-" * 80 + "\n"
        
        # =============================================================================
        # PROMPTY EXECUTION - Load and prepare the prompt using the prompty file
        # =============================================================================
        
        try:
            # Load the prompty file
            prompt_template = load(self.prompty_file_path)
            
            # Prepare the prompt with document data
            prepared_prompt = prepare(prompt_template, documents=document_content)
            
            return prepared_prompt
            
        except Exception as e:
            # Fallback to a basic prompt if prompty fails
            return FALLBACK_PROMPT_TEMPLATE.format(document_content=document_content)
    
    def create_system_message(self) -> str:
        """
        Create the system message for the LLM using Prompty metadata
        
        Returns:
            System message string
        """
        try:
            # Load the prompty file to extract system message
            prompt_template = load(self.prompty_file_path)
            
            # Extract the system message from the prompty template
            # The system message is typically in the 'system' section of the prompty file
            if hasattr(prompt_template, 'system') and prompt_template.system:
                return prompt_template.system.strip()
            else:
                # Fallback system message
                return FALLBACK_SYSTEM_MESSAGE
                
        except Exception as e:
            # Fallback system message if prompty fails
            return FALLBACK_SYSTEM_MESSAGE
