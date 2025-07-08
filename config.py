"""
Configuration settings for Query Fan-Out SEO Analyzer
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # App settings
    APP_NAME = "Query Fan-Out SEO Analyzer"
    APP_VERSION = "1.0.0"
    
    # API Keys (from environment variables)
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
    GSC_CREDENTIALS_PATH = os.getenv('GSC_CREDENTIALS_PATH', '')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    
    # Analysis settings
    DEFAULT_BATCH_SIZE = 50
    MAX_BATCH_SIZE = 100
    MIN_WORD_COUNT = 300
    IDEAL_WORD_COUNT = 1500
    MAX_CONCURRENT_REQUESTS = 10
    REQUEST_TIMEOUT = 30  # seconds
    
    # Rate limiting
    RATE_LIMIT_REQUESTS = 10
    RATE_LIMIT_WINDOW = 1  # seconds
    GSC_RATE_LIMIT_DELAY = 0.1  # 100ms between GSC requests
    
    # Content analysis thresholds
    MIN_CONTENT_SCORE = 0.5
    MIN_READABILITY_SCORE = 60  # Flesch Reading Ease
    MAX_READABILITY_GRADE = 12  # Flesch-Kincaid Grade Level
    
    # Clustering settings
    CLUSTERING_METHOD = 'dbscan'  # or 'kmeans'
    DBSCAN_EPS = 0.3
    DBSCAN_MIN_SAMPLES = 2
    KMEANS_MAX_CLUSTERS = 10
    
    # Query prediction settings
    MAX_PREDICTED_QUERIES = 10
    QUERY_VARIATION_TYPES = [
        'what is', 'how to', 'best', 'guide', 
        'tips', 'tutorial', 'examples', 'benefits'
    ]
    
    # Cache settings
    CACHE_TTL = 3600  # 1 hour
    ENABLE_CACHING = True
    
    # Database settings (if using)
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///query_fanout.db')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    # Export settings
    EXPORT_FORMATS = ['CSV', 'JSON', 'Excel']
    MAX_EXPORT_ROWS = 10000
    
    # UI settings
    SIDEBAR_WIDTH = 300
    MAX_URL_DISPLAY = 20
    CHART_HEIGHT = 400
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Security
    ALLOWED_DOMAINS = []  # Empty list allows all domains
    MAX_URL_LENGTH = 2048
    
    @classmethod
    def get_env_bool(cls, key: str, default: bool = False) -> bool:
        """Get boolean from environment variable"""
        value = os.getenv(key, '').lower()
        return value in ('true', '1', 'yes', 'on') if value else default
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        
        # Check required API keys
        if not cls.GOOGLE_API_KEY and not cls.GSC_CREDENTIALS_PATH:
            errors.append("No Google API credentials configured")
        
        # Validate numeric settings
        if cls.DEFAULT_BATCH_SIZE > cls.MAX_BATCH_SIZE:
            errors.append("DEFAULT_BATCH_SIZE cannot exceed MAX_BATCH_SIZE")
        
        if cls.MIN_WORD_COUNT >= cls.IDEAL_WORD_COUNT:
            errors.append("MIN_WORD_COUNT must be less than IDEAL_WORD_COUNT")
        
        return errors

# Scoring weights for different factors
SCORING_WEIGHTS = {
    'content_length': 0.20,
    'structure': 0.25,
    'technical': 0.15,
    'readability': 0.20,
    'semantic': 0.20
}

# Content structure recommendations
CONTENT_RECOMMENDATIONS = {
    'word_count': {
        'min': 1000,
        'ideal': 1500,
        'max': 3000
    },
    'paragraphs': {
        'min': 10,
        'ideal': 15,
        'max': 30
    },
    'headings': {
        'h1': 1,
        'h2_min': 3,
        'h2_ideal': 5,
        'h3_h6_min': 2
    },
    'lists': {
        'min': 1,
        'ideal': 2
    },
    'images': {
        'min': 1,
        'ideal': 3
    }
}

# SEO best practices
SEO_BEST_PRACTICES = {
    'title_length': {
        'min': 30,
        'ideal': 50,
        'max': 60
    },
    'meta_description_length': {
        'min': 120,
        'ideal': 150,
        'max': 160
    },
    'url_length': {
        'max': 75
    },
    'keyword_density': {
        'min': 0.5,
        'max': 2.5
    }
}

# Query patterns for fan-out analysis
QUERY_PATTERNS = {
    'informational': [
        'what is {keyword}',
        'how does {keyword} work',
        '{keyword} definition',
        '{keyword} meaning',
        'understanding {keyword}'
    ],
    'navigational': [
        '{keyword} website',
        '{keyword} official',
        '{keyword} login',
        '{keyword} app'
    ],
    'transactional': [
        'buy {keyword}',
        '{keyword} price',
        '{keyword} cost',
        'cheap {keyword}',
        '{keyword} deals'
    ],
    'commercial': [
        'best {keyword}',
        '{keyword} reviews',
        '{keyword} comparison',
        'top {keyword}',
        '{keyword} alternatives'
    ],
    'how_to': [
        'how to {keyword}',
        '{keyword} tutorial',
        '{keyword} guide',
        '{keyword} steps',
        '{keyword} instructions'
    ],
    'local': [
        '{keyword} near me',
        '{keyword} in {location}',
        'local {keyword}',
        '{keyword} services'
    ]
}

# Error messages
ERROR_MESSAGES = {
    'invalid_url': 'Invalid URL format. Please check the URL and try again.',
    'connection_error': 'Unable to connect to the URL. Please check your internet connection.',
    'timeout': 'Request timed out. The server took too long to respond.',
    'rate_limit': 'Rate limit exceeded. Please wait before making more requests.',
    'auth_failed': 'Authentication failed. Please check your API credentials.',
    'no_content': 'No content found on the page.',
    'parsing_error': 'Error parsing page content.'
}

# Success messages
SUCCESS_MESSAGES = {
    'analysis_complete': 'Analysis completed successfully!',
    'export_ready': 'Export file is ready for download.',
    'settings_saved': 'Settings saved successfully.',
    'auth_success': 'Authentication successful!'
}
