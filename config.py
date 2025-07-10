"""
Configuration management for Query Fan-Out Analysis Tool
"""

import os
import streamlit as st

class Config:
    """Configuration management for the app"""
    
    @staticmethod
    def get_redirect_uri():
        """Get the appropriate redirect URI based on environment"""
        # First, check if explicitly set in secrets
        try:
            if 'REDIRECT_URI' in st.secrets:
                return st.secrets['REDIRECT_URI']
        except:
            pass
        
        # Check multiple environment variables that Streamlit Cloud sets
        streamlit_cloud_indicators = [
            'STREAMLIT_SHARING_MODE',
            'STREAMLIT_SERVER_HEADLESS',
            'STREAMLIT_RUNTIME_ENV'
        ]
        
        # If any of these are set, we're on Streamlit Cloud
        if any(var in os.environ for var in streamlit_cloud_indicators):
            # Get the app name from secrets or use default
            try:
                app_name = st.secrets.get('APP_NAME', 'seo-query-fan-out-tool')
            except:
                app_name = 'seo-query-fan-out-tool'
            
            return f'https://{app_name}.streamlit.app'
        
        # Local development
        return 'http://localhost:8501'
    
    @staticmethod
    def get_google_credentials():
        """Get Google OAuth credentials"""
        client_id = None
        client_secret = None
        
        # Try secrets first
        try:
            if 'GOOGLE_CLIENT_ID' in st.secrets:
                client_id = st.secrets['GOOGLE_CLIENT_ID']
                client_secret = st.secrets['GOOGLE_CLIENT_SECRET']
        except:
            pass
        
        # Try environment variables
        if not client_id:
            client_id = os.getenv('GOOGLE_CLIENT_ID')
            client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
        
        return client_id, client_secret
    
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
    
    # OAuth scopes
    SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']
    
    # Analysis defaults
    DEFAULT_DAYS_BACK = 30
    DEFAULT_MIN_CLICKS = 5
    DEFAULT_MAX_QUERIES = 20
