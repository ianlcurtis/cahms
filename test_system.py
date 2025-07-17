#!/usr/bin/env python3
"""
Test script to validate CAHMS system configuration and functionality.
"""
import os
import sys
from dotenv import load_dotenv

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_environment_variables():
    """Test that all required environment variables are loaded correctly."""
    load_dotenv()
    
    # Test basic configuration
    required_vars = ['TITLE', 'LLM_ENDPOINT', 'LLM_API_KEY', 'LLM_MODEL_NAME']
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value[:20]}{'...' if len(value) > 20 else ''}")
        else:
            print(f"❌ {var}: Not set")
    
    # Test mandatory upload configuration
    mandatory_uploads = [
        'MANDATORY_FORM_S',
        'MANDATORY_FORM_H',
        'MANDATORY_FORM_A',
        'MANDATORY_CAHMS_INITIAL',
        'MANDATORY_NEURO_DEV_HISTORY',
        'MANDATORY_FORMULATION_DOCUMENT',
        'MANDATORY_SCHOOL_OBSERVATION',
        'MANDATORY_SUPPORTING_INFO'
    ]
    
    print("\n📋 Mandatory Upload Configuration:")
    for var in mandatory_uploads:
        value = os.getenv(var, 'false').lower() == 'true'
        status = "Required" if value else "Optional"
        field_name = var.replace('MANDATORY_', '').replace('_', ' ').title()
        print(f"  {field_name}: {status}")

def test_imports():
    """Test that all required modules can be imported."""
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
    
    try:
        from azure_llm_client import AzureLLMClient
        print("✅ Azure LLM Client imported successfully")
    except ImportError as e:
        print(f"❌ Azure LLM Client import failed: {e}")
    
    try:
        import openai
        print("✅ OpenAI SDK imported successfully")
    except ImportError as e:
        print(f"❌ OpenAI SDK import failed: {e}")

def test_llm_client():
    """Test that the LLM client can be instantiated."""
    try:
        from azure_llm_client import AzureLLMClient
        client = AzureLLMClient()
        print("✅ Azure LLM Client instantiated successfully")
        return True
    except Exception as e:
        print(f"❌ Azure LLM Client instantiation failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 CAHMS System Test")
    print("=" * 50)
    
    print("\n1. Testing Environment Variables:")
    test_environment_variables()
    
    print("\n2. Testing Imports:")
    test_imports()
    
    print("\n3. Testing LLM Client:")
    test_llm_client()
    
    print("\n✅ Test completed! If all checks passed, the system is ready.")
    print("\n🚀 To run the application:")
    print("   streamlit run src/Home.py")
