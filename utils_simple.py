"""
Utility functions for Query Fan-Out SEO Analyzer
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urljoin
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, KMeans
import spacy
from collections import Counter
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd

# Load spaCy model (download with: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class ContentAnalyzer:
    """Advanced content analysis for SEO optimization"""
    
    def __init__(self):
        self.sentence_model = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Simple entity extraction using regex patterns"""
        entities = []
        
        # Extract capitalized words as potential entities
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(capitalized_pattern, text)
        
        for match in matches:
            if len(match) > 2:  # Skip short matches
                entities.append({
                    'text': match,
                    'label': 'ENTITY',
                    'start': text.find(match),
                    'end': text.find(match) + len(match)
                })
        
        return entities[:20]  # Limit to top 20 entities
    
    def calculate_readability_scores(self, text: str) -> Dict[str, float]:
        """Calculate basic readability scores"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        words = text.split()
        
        # Basic metrics
        num_sentences = len(sentences)
        num_words = len(words)
        
        if num_sentences == 0 or num_words == 0:
            return {
                'avg_words_per_sentence': 0,
                'avg_word_length': 0,
                'complexity_score': 0
            }
        
        # Calculate averages
        avg_words_per_sentence = num_words / num_sentences
        avg_word_length = sum(len(word) for word in words) / num_words
        
        # Simple complexity score
        long_words = len([w for w in words if len(w) > 6])
        complexity_score = (long_words / num_words) * 100 if num_words > 0 else 0
        
        return {
            'avg_words_per_sentence': avg_words_per_sentence,
            'avg_word_length': avg_word_length,
            'complexity_score': complexity_score
        }
    
    def extract_content_chunks(self, html_content: str) -> List[Dict[str, Any]]:
        """Extract content chunks from HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        chunks = []
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract main content areas
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = tag.get_text(strip=True)
            if len(text) > 20:  # Minimum chunk size
                chunks.append({
                    'type': tag.name,
                    'text': text,
                    'word_count': len(text.split()),
                    'parent': tag.parent.name if tag.parent else None
                })
        
        return chunks
    
    def calculate_semantic_similarity(self, texts: List[str]) -> np.ndarray:
        """Calculate semantic similarity between texts"""
        if not self.sentence_model:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        embeddings = self.sentence_model.encode(texts)
        similarity_matrix = cosine_similarity(embeddings)
        
        return similarity_matrix
    
    def cluster_content(self, texts: List[str], method: str = 'dbscan') -> List[int]:
        """Cluster content using semantic similarity"""
        if not texts:
            return []
        
        # Get embeddings
        if not self.sentence_model:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        embeddings = self.sentence_model.encode(texts)
        
        if method == 'dbscan':
            clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
            labels = clustering.fit_predict(embeddings)
        else:  # kmeans
            n_clusters = min(5, len(texts) // 10)  # Adaptive cluster count
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clustering.fit_predict(embeddings)
        
        return labels.tolist()
    
    def generate_query_variations(self, base_query: str, variation_types: List[str]) -> List[str]:
        """Generate query variations based on patterns"""
        variations = []
        base_query_lower = base_query.lower().strip()
        
        variation_patterns = {
            'what is': f"what is {base_query_lower}",
            'how to': f"how to {base_query_lower}",
            'best': f"best {base_query_lower}",
            'guide': f"{base_query_lower} guide",
            'tips': f"{base_query_lower} tips",
            'tutorial': f"{base_query_lower} tutorial",
            'examples': f"{base_query_lower} examples",
            'benefits': f"benefits of {base_query_lower}",
            'vs': f"{base_query_lower} vs",
            'review': f"{base_query_lower} review"
        }
        
        for var_type in variation_types:
            if var_type.lower() in variation_patterns:
                variations.append(variation_patterns[var_type.lower()])
        
        return variations

class SEOScorer:
    """Calculate SEO scores for content"""
    
    def __init__(self):
        self.ideal_word_count = 1500
        self.ideal_paragraph_count = 15
        self.ideal_heading_count = 8
    
    def calculate_comprehensive_score(self, content_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive SEO score with breakdown"""
        scores = {
            'content_length_score': self._score_content_length(content_data.get('word_count', 0)),
            'structure_score': self._score_structure(content_data),
            'technical_score': self._score_technical(content_data),
            'readability_score': 0.7,  # Default score
            'semantic_score': self._score_semantic_coverage(content_data)
        }
        
        # Calculate overall score (weighted average)
        weights = {
            'content_length_score': 0.2,
            'structure_score': 0.25,
            'technical_score': 0.15,
            'readability_score': 0.2,
            'semantic_score': 0.2
        }
        
        overall_score = sum(scores[key] * weights[key] for key in scores)
        scores['overall_score'] = overall_score
        
        return scores
    
    def _score_content_length(self, word_count: int) -> float:
        """Score based on content length"""
        if word_count >= self.ideal_word_count:
            return 1.0
        elif word_count >= 1000:
            return 0.8 + (word_count - 1000) / 2500
        elif word_count >= 500:
            return 0.5 + (word_count - 500) / 1000
        else:
            return word_count / 1000
    
    def _score_structure(self, content_data: Dict[str, Any]) -> float:
        """Score based on content structure"""
        score = 0.0
        
        # Heading structure
        h1_count = content_data.get('h1_count', 0)
        h2_count = content_data.get('h2_count', 0)
        
        if h1_count == 1:
            score += 0.2
        if h2_count >= 5:
            score += 0.3
        elif h2_count >= 3:
            score += 0.2
        
        # Paragraph structure
        paragraph_count = content_data.get('paragraph_count', 0)
        if paragraph_count >= self.ideal_paragraph_count:
            score += 0.3
        else:
            score += 0.3 * (paragraph_count / self.ideal_paragraph_count)
        
        # Lists
        if content_data.get('list_count', 0) >= 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_technical(self, content_data: Dict[str, Any]) -> float:
        """Score technical SEO factors"""
        score = 0.0
        
        # Title tag
        title_length = len(content_data.get('title', ''))
        if 30 <= title_length <= 60:
            score += 0.3
        elif title_length > 0:
            score += 0.15
        
        # Meta description
        meta_length = len(content_data.get('meta_description', ''))
        if 120 <= meta_length <= 160:
            score += 0.3
        elif meta_length > 0:
            score += 0.15
        
        # Schema markup
        if content_data.get('has_schema', False):
            score += 0.4
        
        return score
    
    def _score_semantic_coverage(self, content_data: Dict[str, Any]) -> float:
        """Score semantic coverage based on predicted queries"""
        predicted_queries = content_data.get('predicted_queries', [])
        if len(predicted_queries) >= 8:
            return 0.9
        elif len(predicted_queries) >= 5:
            return 0.7
        else:
            return 0.5

class URLProcessor:
    """Process and validate URLs"""
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize URL for consistency"""
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Remove trailing slash
        url = url.rstrip('/')
        
        # Parse and reconstruct
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        
        return normalized
    
    @staticmethod
    def get_domain(url: str) -> str:
        """Extract domain from URL"""
        parsed = urlparse(url)
        return parsed.netloc
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if URL is valid"""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc) and bool(parsed.scheme)
        except:
            return False

class BatchProcessor:
    """Handle batch processing with rate limiting"""
    
    def __init__(self, rate_limit: int = 10, time_window: int = 1):
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.semaphore = asyncio.Semaphore(rate_limit)
    
    async def process_batch(self, items: List[Any], process_func, *args, **kwargs) -> List[Any]:
        """Process items in batch with rate limiting"""
        async def process_with_limit(item):
            async with self.semaphore:
                result = await process_func(item, *args, **kwargs)
                await asyncio.sleep(self.time_window / self.rate_limit)
                return result
        
        tasks = [process_with_limit(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results

def export_to_excel(dataframe: pd.DataFrame, filename: str) -> bytes:
    """Export DataFrame to Excel with formatting"""
    from io import BytesIO
    
    output = BytesIO()
    
    # Use xlsxwriter engine
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write dataframe
        dataframe.to_excel(writer, sheet_name='Analysis Results', index=False)
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Analysis Results']
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4CAF50',
            'font_color': 'white',
            'border': 1
        })
        
        # Format headers
        for col_num, value in enumerate(dataframe.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Auto-fit columns
        for i, col in enumerate(dataframe.columns):
            column_width = max(dataframe[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, min(column_width, 50))  # Max width 50
    
    output.seek(0)
    return output.read()

def generate_recommendations(analysis_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate actionable SEO recommendations based on analysis"""
    recommendations = []
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(analysis_results)
    success_df = df[df['status'] == 'success']
    
    if success_df.empty:
        return recommendations
    
    # Content length recommendations
    avg_word_count = success_df['word_count'].mean()
    if avg_word_count < 1000:
        recommendations.append({
            'category': 'Content Length',
            'priority': 'High',
            'issue': f'Average word count is {avg_word_count:.0f}',
            'recommendation': 'Expand content to at least 1000-1500 words for better topical coverage',
            'impact': 'High impact on query coverage and ranking potential'
        })
    
    # Structure recommendations
    low_structure_pages = success_df[success_df['h2_count'] < 3]
    if len(low_structure_pages) > len(success_df) * 0.3:
        recommendations.append({
            'category': 'Content Structure',
            'priority': 'Medium',
            'issue': f'{len(low_structure_pages)} pages lack proper heading structure',
            'recommendation': 'Add more H2 subheadings to improve content hierarchy',
            'impact': 'Improves user experience and content scanability'
        })
    
    # Schema recommendations
    no_schema_pages = success_df[~success_df['has_schema']]
    if len(no_schema_pages) > 0:
        recommendations.append({
            'category': 'Technical SEO',
            'priority': 'Medium',
            'issue': f'{len(no_schema_pages)} pages lack schema markup',
            'recommendation': 'Implement appropriate schema.org markup',
            'impact': 'Enhances rich snippet eligibility and search visibility'
        })
    
    # Coverage score recommendations
    low_coverage = success_df[success_df.get('coverage_score', 0) < 0.5]
    if len(low_coverage) > 0:
        recommendations.append({
            'category': 'Content Quality',
            'priority': 'High',
            'issue': f'{len(low_coverage)} pages have low coverage scores',
            'recommendation': 'Improve content comprehensiveness and topical coverage',
            'impact': 'Critical for query fan-out optimization'
        })
    
    return recommendations
