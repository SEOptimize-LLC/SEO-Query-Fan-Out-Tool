"""
Simplified utility functions for Query Fan-Out SEO Analyzer
Works without heavy ML dependencies but keeps core functionality
"""

import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import numpy as np
from collections import Counter
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd

class ContentAnalyzer:
    """Simplified content analysis for SEO optimization"""
    
    def __init__(self):
        self.stop_words = set(['the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'some', 'any', 'few', 'many', 'much', 'most', 'other', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once'])
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Simple entity extraction using capitalized words"""
        entities = []
        
        # Find capitalized sequences (potential entities)
        words = text.split()
        i = 0
        while i < len(words):
            if words[i][0].isupper() and words[i].lower() not in self.stop_words:
                entity = words[i]
                j = i + 1
                # Collect consecutive capitalized words
                while j < len(words) and words[j][0].isupper():
                    entity += ' ' + words[j]
                    j += 1
                
                if len(entity) > 2:  # Skip single letters
                    entities.append({
                        'text': entity,
                        'label': 'ENTITY',
                        'start': text.find(entity),
                        'end': text.find(entity) + len(entity)
                    })
                i = j
            else:
                i += 1
        
        return entities[:20]  # Return top 20 entities
    
    def calculate_readability_scores(self, text: str) -> Dict[str, float]:
        """Calculate basic readability metrics"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words = text.split()
        
        if not sentences or not words:
            return {
                'avg_words_per_sentence': 0,
                'avg_word_length': 0,
                'complexity_score': 0
            }
        
        avg_words_per_sentence = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple complexity score based on long words
        long_words = len([w for w in words if len(w) > 6])
        complexity_score = (long_words / len(words)) * 100
        
        return {
            'avg_words_per_sentence': round(avg_words_per_sentence, 2),
            'avg_word_length': round(avg_word_length, 2),
            'complexity_score': round(complexity_score, 2)
        }
    
    def extract_content_chunks(self, html_content: str) -> List[Dict[str, Any]]:
        """Extract content chunks from HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        chunks = []
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text from different elements
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
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        url = url.rstrip('/')
        
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
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        dataframe.to_excel(writer, sheet_name='Analysis Results', index=False)
        
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
            worksheet.set_column(i, i, min(column_width, 50))
    
    output.seek(0)
    return output.read()

def generate_recommendations(analysis_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate actionable SEO recommendations based on analysis"""
    recommendations = []
    
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
    
    return recommendations
