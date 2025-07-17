"""
Assessment Prompt Generator for CAHMS Neurodevelopmental Assessment
This module handles the creation of prompts for LLM-based assessment report generation.
"""

from typing import List, Any


class AssessmentPromptGenerator:
    """Generates prompts for neurodevelopmental assessment reports"""
    
    def create_assessment_prompt(self, documents: List[Any]) -> str:
        """
        Create a comprehensive prompt for the LLM based on uploaded documents
        
        Args:
            documents: List of processed documents
            
        Returns:
            Formatted prompt for the LLM
        """
        
        # =============================================================================
        # SYSTEM PROMPT - Edit this section to modify the base instructions and output format
        # =============================================================================
        
        system_prompt = """You are a specialist clinician working with the CAMHS (Child and Adolescent Mental Health Services) Neurodevelopmental Team. 

Your task is to analyze the provided assessment documents and generate a comprehensive neurodevelopmental assessment report.

ASSESSMENT TASK:
Please analyze all the provided documents and generate a comprehensive neurodevelopmental assessment report that includes:

1. **Executive Summary**
   - Key findings and recommendations
   - Primary neurodevelopmental concerns identified

2. **Background Information**
   - Relevant developmental history
   - Family context and environmental factors
   - Educational background and challenges

3. **Assessment Findings**
   - Analysis of attention, concentration, and hyperactivity (Form H)
   - Social interaction and communication assessment (Form A)
   - School-based observations and support needs (Form S)
   - Clinical observations from initial assessment

4. **Formulation**
   - Integration of all assessment information
   - Identified strengths and challenges
   - Neurodevelopmental profile

5. **Recommendations**
   - Intervention strategies
   - Support recommendations for school and home
   - Follow-up requirements
   - Referrals to other services if needed

6. **Conclusion**
   - Summary of key points
   - Prognosis and next steps

Please ensure the report is:
- Professional and clinical in tone
- Evidence-based using the provided documentation
- Comprehensive yet accessible
- Structured with clear headings and bullet points where appropriate
- Sensitive to the young person and family's needs

Generate the assessment report now:
"""
        
        # =============================================================================
        # DOCUMENT CONTENT INSERTION - This section is automatically populated
        # =============================================================================
        
        document_section = "\n\nAVAILABLE DOCUMENTS:\n"
        
        for doc in documents:
            status = "REQUIRED" if doc.is_required else "OPTIONAL"
            document_section += f"\n{status} - {doc.document_type} ({doc.filename}):\n"
            document_section += f"{doc.content}\n"
            document_section += "-" * 80 + "\n"
        
        # =============================================================================
        # FINAL PROMPT ASSEMBLY - Combines all sections
        # =============================================================================
        
        final_prompt = system_prompt + document_section
        
        return final_prompt
    
    def create_system_message(self) -> str:
        """
        Create the system message for the LLM
        
        Returns:
            System message string
        """
        return "You are an expert clinician specializing in neurodevelopmental assessments for children and adolescents."
