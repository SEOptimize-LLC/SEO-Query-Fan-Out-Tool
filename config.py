"""
Configuration management for Query Fan-Out Analysis Tool
"""

import os
import streamlit as st

class Config:
    """Configuration management for the app"""
    
    @staticmethod
    def get_gemini_api_key():
        """Get Gemini API key"""
        # Try secrets first
        try:
            if 'GEMINI_API_KEY' in st.secrets:
                return st.secrets['GEMINI_API_KEY']
        except:
            pass
        
        # Try environment variable
        return os.getenv('GEMINI_API_KEY')
    
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
    MIN_CONTENT_LENGTH = 100  # Minimum words for valid content
    MAX_CONTENT_LENGTH = 50000  # Maximum words to analyze
    
    # User Agent for web scraping
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
