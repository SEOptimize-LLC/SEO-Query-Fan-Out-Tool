"""
Configuration management for Query Fan-Out Analysis Tool
"""

import os
import streamlit as st

class Config:
    """Configuration management for the app"""
    
    @staticmethod
    def get_gemini_api_key():
        """Get Gemini API key from secrets or environment"""
        try:
            if 'GEMINI_API_KEY' in st.secrets:
                return st.secrets['GEMINI_API_KEY']
        except:
            pass
        return os.getenv('GEMINI_API_KEY', '')
    
    @staticmethod
    def get_openai_api_key():
        """Get OpenAI API key from secrets or environment"""
        try:
            if 'OPENAI_API_KEY' in st.secrets:
                return st.secrets['OPENAI_API_KEY']
        except:
            pass
        return os.getenv('OPENAI_API_KEY', '')
    
    @staticmethod
    def get_anthropic_api_key():
        """Get Anthropic API key from secrets or environment"""
        try:
            if 'ANTHROPIC_API_KEY' in st.secrets:
                return st.secrets['ANTHROPIC_API_KEY']
        except:
            pass
        return os.getenv('ANTHROPIC_API_KEY', '')
    
    @staticmethod
    def get_all_api_keys():
        """Get all configured API keys"""
        return {
            'gemini': Config.get_gemini_api_key(),
            'openai': Config.get_openai_api_key(),
            'anthropic': Config.get_anthropic_api_key()
        }
    
    # Model configurations - focused on best reasoning/thinking models
    AI_MODELS = {
        'gemini': {
            'name': 'Google Gemini',
            'models': [
                {'id': 'gemini-1.5-pro', 'name': 'Gemini 1.5 Pro', 'description': 'Best for complex analysis and reasoning'},
                {'id': 'gemini-1.5-flash', 'name': 'Gemini 1.5 Flash', 'description': 'Faster, good for simpler tasks'},
                {'id': 'gemini-exp-1206', 'name': 'Gemini Experimental', 'description': 'Latest experimental features'},
            ]
        },
        'openai': {
            'name': 'OpenAI',
            'models': [
                {'id': 'o1', 'name': 'O1 (Best Reasoning)', 'description': 'Most advanced reasoning and thinking model'},
                {'id': 'o1-preview', 'name': 'O1 Preview', 'description': 'Preview version with strong reasoning'},
                {'id': 'o1-mini', 'name': 'O1 Mini', 'description': 'Faster reasoning model'},
                {'id': 'gpt-4o', 'name': 'GPT-4o', 'description': 'Latest GPT-4, fast and capable'},
                {'id': 'gpt-4o-mini', 'name': 'GPT-4o Mini', 'description': 'Smaller, faster, more affordable'},
            ]
        },
        'anthropic': {
            'name': 'Anthropic Claude',
            'models': [
                {'id': 'claude-3-5-sonnet-20241022', 'name': 'Claude 3.5 Sonnet', 'description': 'Best overall - excellent reasoning'},
                {'id': 'claude-3-opus-20240229', 'name': 'Claude 3 Opus', 'description': 'Most capable for complex tasks'},
                {'id': 'claude-3-haiku-20240307', 'name': 'Claude 3 Haiku', 'description': 'Fast and efficient'},
            ]
        }
    }
    
    # Analysis defaults
    DEFAULT_MAX_QUERIES = 20
    DEFAULT_MAX_TOPICS = 15
    DEFAULT_ANALYSIS_DEPTH = "Standard"
    
    # Query Fan-Out Variant Types (based on Google's system)
    VARIANT_TYPES = {
        'equivalent': 'Alternative ways to ask the same question',
        'follow_up': 'Logical next questions that build on the original',
        'generalization': 'Broader versions of the specific question',
        'canonicalization': 'Standardized or normalized versions',
        'entailment': 'Queries that logically follow from the original',
        'specification': 'More detailed or specific versions',
        'clarification': 'Questions to clarify user intent'
    }
    
    # Content Analysis Settings
    MIN_CONTENT_LENGTH = 100
    MAX_CONTENT_LENGTH = 50000
    
    # User Agent for web scraping
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
