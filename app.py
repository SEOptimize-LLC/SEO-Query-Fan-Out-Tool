import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import requests
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import json
from bs4 import BeautifulSoup

# Page configuration
st.set_page_config(
    page_title="Query Fan-Out SEO Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'gsc_data' not in st.session_state:
    st.session_state.gsc_data = None
if 'sitemap_urls' not in st.session_state:
    st.session_state.sitemap_urls = []

class QueryFanOutAnalyzer:
    """Main class for Query Fan-Out SEO Analysis"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    async def fetch_sitemap_urls(self, sitemap_url: str) -> List[str]:
        """Fetch all URLs from a sitemap"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(sitemap_url, headers=self.headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self._parse_sitemap(content)
        except Exception as e:
            st.error(f"Error fetching sitemap: {str(e)}")
            return []
    
    def _parse_sitemap(self, content: str) -> List[str]:
        """Parse sitemap XML and extract URLs"""
        urls = []
        try:
            root = ET.fromstring(content)
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            
            # Check if it's a sitemap index
            sitemaps = root.findall('.//ns:sitemap/ns:loc', namespace)
            if sitemaps:
                for sitemap in sitemaps:
                    # Recursively fetch URLs from child sitemaps
                    child_content = requests.get(sitemap.text).text
                    urls.extend(self._parse_sitemap(child_content))
            else:
                # Regular sitemap with URLs
                url_elements = root.findall('.//ns:url/ns:loc', namespace)
                urls = [url.text for url in url_elements]
        except Exception as e:
            st.error(f"Error parsing sitemap: {str(e)}")
        
        return urls
    
    async def analyze_content(self, url: str) -> Dict[str, Any]:
        """Analyze a single URL for query fan-out optimization"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self._analyze_page_content(url, content)
        except Exception as e:
            return {
                'url': url,
                'status': 'error',
                'error': str(e)
            }
    
    def _analyze_page_content(self, url: str, content: str) -> Dict[str, Any]:
        """Extract and analyze page content"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract basic elements
        title = soup.find('title')
        meta_desc = soup.find('meta', {'name': 'description'})
        h1_tags = soup.find_all('h1')
        h2_tags = soup.find_all('h2')
        
        # Extract main content
        main_content = soup.get_text(separator=' ', strip=True)
        word_count = len(main_content.split())
        
        # Content structure analysis
        paragraphs = soup.find_all('p')
        lists = soup.find_all(['ul', 'ol'])
        
        # Schema markup detection
        schema_scripts = soup.find_all('script', {'type': 'application/ld+json'})
        has_schema = len(schema_scripts) > 0
        
        return {
            'url': url,
            'status': 'success',
            'title': title.text if title else '',
            'meta_description': meta_desc.get('content', '') if meta_desc else '',
            'h1_count': len(h1_tags),
            'h2_count': len(h2_tags),
            'word_count': word_count,
            'paragraph_count': len(paragraphs),
            'list_count': len(lists),
            'has_schema': has_schema,
            'content_preview': main_content[:500] + '...' if len(main_content) > 500 else main_content
        }
    
    def generate_query_predictions(self, content_data: Dict[str, Any]) -> List[str]:
        """Generate predicted sub-queries based on content"""
        # Simplified query prediction logic
        # In production, this would use NLP models
        predicted_queries = []
        
        title = content_data.get('title', '')
        if title:
            # Generate variations
            predicted_queries.append(f"what is {title.lower()}")
            predicted_queries.append(f"how to {title.lower()}")
            predicted_queries.append(f"{title.lower()} guide")
            predicted_queries.append(f"{title.lower()} tips")
            predicted_queries.append(f"best {title.lower()}")
        
        return predicted_queries[:10]  # Limit to 10 queries
    
    def calculate_coverage_score(self, content_data: Dict[str, Any]) -> float:
        """Calculate content coverage score"""
        score = 0.0
        
        # Basic scoring factors
        if content_data.get('word_count', 0) > 1000:
            score += 0.2
        if content_data.get('h2_count', 0) >= 5:
            score += 0.2
        if content_data.get('has_schema', False):
            score += 0.1
        if content_data.get('paragraph_count', 0) >= 10:
            score += 0.2
        if content_data.get('list_count', 0) >= 2:
            score += 0.1
        if len(content_data.get('meta_description', '')) > 100:
            score += 0.2
        
        return min(score, 1.0)

# Sidebar configuration
st.sidebar.title("🎯 Query Fan-Out Analyzer")
st.sidebar.markdown("---")

# Input method selection
input_method = st.sidebar.radio(
    "Select Input Method",
    ["Sitemap URL", "Upload CSV", "Manual URLs"]
)

# GSC API Configuration (placeholder)
st.sidebar.markdown("### Google Search Console")
gsc_api_key = st.sidebar.text_input("API Key", type="password", help="Enter your GSC API key")
gsc_property = st.sidebar.text_input("Property URL", placeholder="https://example.com")

# Analysis settings
st.sidebar.markdown("### Analysis Settings")
batch_size = st.sidebar.slider("Batch Size", 10, 100, 50)
enable_clustering = st.sidebar.checkbox("Enable Semantic Clustering", value=True)
export_format = st.sidebar.selectbox("Export Format", ["CSV", "JSON", "Excel"])

# Main content area
st.title("🎯 Query Fan-Out SEO Analyzer")
st.markdown("Optimize your content for Google's AI-powered search using query fan-out analysis")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "📈 Results", "🔍 Insights", "⚙️ Settings"])

with tab1:
    st.header("Content Analysis")
    
    analyzer = QueryFanOutAnalyzer()
    
    # Input handling based on method
    if input_method == "Sitemap URL":
        sitemap_url = st.text_input("Enter Sitemap URL", placeholder="https://example.com/sitemap.xml")
        
        if st.button("Fetch URLs from Sitemap", type="primary"):
            if sitemap_url:
                with st.spinner("Fetching URLs from sitemap..."):
                    # Run async function
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    urls = loop.run_until_complete(analyzer.fetch_sitemap_urls(sitemap_url))
                    
                    if urls:
                        st.session_state.sitemap_urls = urls
                        st.success(f"Found {len(urls)} URLs in sitemap")
                        
                        # Display URL preview
                        with st.expander("Preview URLs"):
                            df_preview = pd.DataFrame(urls[:20], columns=['URL'])
                            st.dataframe(df_preview)
    
    elif input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if 'url' in df.columns or 'URL' in df.columns:
                url_column = 'url' if 'url' in df.columns else 'URL'
                st.session_state.sitemap_urls = df[url_column].tolist()
                st.success(f"Loaded {len(st.session_state.sitemap_urls)} URLs from CSV")
    
    else:  # Manual URLs
        urls_text = st.text_area("Enter URLs (one per line)", height=200)
        
        if st.button("Load URLs"):
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            if urls:
                st.session_state.sitemap_urls = urls
                st.success(f"Loaded {len(urls)} URLs")
    
    # Analysis section
    if st.session_state.sitemap_urls:
        st.markdown("---")
        st.subheader("Ready to Analyze")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total URLs", len(st.session_state.sitemap_urls))
        with col2:
            estimated_time = (len(st.session_state.sitemap_urls) / batch_size) * 30  # seconds
            st.metric("Estimated Time", f"{estimated_time/60:.1f} min")
        with col3:
            st.metric("Batch Size", batch_size)
        
        if st.button("🚀 Start Analysis", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            total_urls = len(st.session_state.sitemap_urls)
            
            # Process URLs in batches
            for i in range(0, total_urls, batch_size):
                batch = st.session_state.sitemap_urls[i:i+batch_size]
                status_text.text(f"Processing batch {i//batch_size + 1} of {(total_urls-1)//batch_size + 1}")
                
                # Analyze each URL in batch
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                tasks = [analyzer.analyze_content(url) for url in batch]
                batch_results = loop.run_until_complete(asyncio.gather(*tasks))
                
                # Add predictions and scores
                for result in batch_results:
                    if result['status'] == 'success':
                        result['predicted_queries'] = analyzer.generate_query_predictions(result)
                        result['coverage_score'] = analyzer.calculate_coverage_score(result)
                
                results.extend(batch_results)
                
                # Update progress
                progress = min((i + batch_size) / total_urls, 1.0)
                progress_bar.progress(progress)
                
                # Small delay to avoid rate limiting
                time.sleep(1)
            
            st.session_state.analysis_results = results
            status_text.text("Analysis complete!")
            st.success(f"Successfully analyzed {len(results)} URLs")

with tab2:
    st.header("Analysis Results")
    
    if st.session_state.analysis_results:
        results_df = pd.DataFrame(st.session_state.analysis_results)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            success_rate = len(results_df[results_df['status'] == 'success']) / len(results_df) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        with col2:
            avg_word_count = results_df[results_df['status'] == 'success']['word_count'].mean()
            st.metric("Avg Word Count", f"{avg_word_count:.0f}")
        
        with col3:
            avg_coverage = results_df[results_df['status'] == 'success']['coverage_score'].mean()
            st.metric("Avg Coverage Score", f"{avg_coverage:.2f}")
        
        with col4:
            schema_rate = results_df[results_df['status'] == 'success']['has_schema'].sum() / len(results_df[results_df['status'] == 'success']) * 100
            st.metric("Schema Markup", f"{schema_rate:.1f}%")
        
        # Visualizations
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Coverage score distribution
            fig_coverage = px.histogram(
                results_df[results_df['status'] == 'success'],
                x='coverage_score',
                nbins=20,
                title='Content Coverage Score Distribution'
            )
            st.plotly_chart(fig_coverage, use_container_width=True)
        
        with col2:
            # Word count distribution
            fig_words = px.box(
                results_df[results_df['status'] == 'success'],
                y='word_count',
                title='Word Count Distribution'
            )
            st.plotly_chart(fig_words, use_container_width=True)
        
        # Detailed results table
        st.markdown("---")
        st.subheader("Detailed Results")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            score_filter = st.slider("Min Coverage Score", 0.0, 1.0, 0.0)
        with col2:
            word_filter = st.number_input("Min Word Count", 0, 5000, 0)
        with col3:
            status_filter = st.selectbox("Status", ["All", "Success", "Error"])
        
        # Apply filters
        filtered_df = results_df.copy()
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df['status'] == status_filter.lower()]
        if status_filter == "Success" or status_filter == "All":
            success_df = filtered_df[filtered_df['status'] == 'success']
            if score_filter > 0:
                success_df = success_df[success_df['coverage_score'] >= score_filter]
            if word_filter > 0:
                success_df = success_df[success_df['word_count'] >= word_filter]
            
            if status_filter == "All":
                error_df = filtered_df[filtered_df['status'] == 'error']
                filtered_df = pd.concat([success_df, error_df])
            else:
                filtered_df = success_df
        
        # Display filtered results
        st.dataframe(
            filtered_df[['url', 'title', 'word_count', 'coverage_score', 'has_schema', 'status']],
            use_container_width=True
        )
        
        # Export options
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"query_fanout_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = filtered_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"query_fanout_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # Excel export would require additional library
            st.info("Excel export requires openpyxl library")
    
    else:
        st.info("No analysis results available. Please run an analysis first.")

with tab3:
    st.header("SEO Insights & Recommendations")
    
    if st.session_state.analysis_results:
        results_df = pd.DataFrame(st.session_state.analysis_results)
        success_df = results_df[results_df['status'] == 'success']
        
        # Low coverage score pages
        st.subheader("🔴 Pages Needing Immediate Attention")
        low_coverage = success_df[success_df['coverage_score'] < 0.5].sort_values('coverage_score')
        
        if not low_coverage.empty:
            for idx, row in low_coverage.head(10).iterrows():
                with st.expander(f"{row['title'][:50]}... (Score: {row['coverage_score']:.2f})"):
                    st.write(f"**URL:** {row['url']}")
                    st.write(f"**Word Count:** {row['word_count']}")
                    st.write(f"**H2 Tags:** {row['h2_count']}")
                    
                    st.markdown("**Recommendations:**")
                    if row['word_count'] < 1000:
                        st.write("- 📝 Expand content to at least 1000 words")
                    if row['h2_count'] < 5:
                        st.write("- 🏷️ Add more H2 subheadings for better structure")
                    if not row['has_schema']:
                        st.write("- 🔧 Implement schema markup")
                    if row['paragraph_count'] < 10:
                        st.write("- 📄 Break content into more paragraphs")
        else:
            st.success("All pages have good coverage scores!")
        
        # Content gaps analysis
        st.markdown("---")
        st.subheader("📊 Content Gap Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pages without schema
            no_schema = success_df[~success_df['has_schema']]
            st.metric("Pages without Schema", len(no_schema))
            
            if len(no_schema) > 0:
                with st.expander("View pages without schema"):
                    st.dataframe(no_schema[['url', 'title']], use_container_width=True)
        
        with col2:
            # Short content pages
            short_content = success_df[success_df['word_count'] < 500]
            st.metric("Pages with <500 words", len(short_content))
            
            if len(short_content) > 0:
                with st.expander("View short content pages"):
                    st.dataframe(short_content[['url', 'title', 'word_count']], use_container_width=True)
        
        # Query coverage insights
        st.markdown("---")
        st.subheader("🎯 Query Coverage Insights")
        
        # Sample predicted queries analysis
        all_queries = []
        for result in st.session_state.analysis_results:
            if result['status'] == 'success' and 'predicted_queries' in result:
                all_queries.extend(result['predicted_queries'])
        
        if all_queries:
            query_counts = pd.Series(all_queries).value_counts().head(20)
            
            fig_queries = px.bar(
                x=query_counts.values,
                y=query_counts.index,
                orientation='h',
                title='Most Common Predicted Queries',
                labels={'x': 'Frequency', 'y': 'Query'}
            )
            st.plotly_chart(fig_queries, use_container_width=True)
        
        # Actionable recommendations
        st.markdown("---")
        st.subheader("💡 Actionable Recommendations")
        
        recommendations = []
        
        # Calculate overall metrics
        avg_word_count = success_df['word_count'].mean()
        avg_coverage = success_df['coverage_score'].mean()
        schema_rate = success_df['has_schema'].sum() / len(success_df) * 100
        
        if avg_word_count < 1000:
            recommendations.append({
                'priority': 'High',
                'area': 'Content Length',
                'recommendation': 'Increase average content length to at least 1000 words for better query coverage'
            })
        
        if avg_coverage < 0.7:
            recommendations.append({
                'priority': 'High',
                'area': 'Content Structure',
                'recommendation': 'Improve content structure with more subheadings, lists, and comprehensive coverage'
            })
        
        if schema_rate < 80:
            recommendations.append({
                'priority': 'Medium',
                'area': 'Technical SEO',
                'recommendation': f'Implement schema markup on {100-schema_rate:.0f}% of pages lacking structured data'
            })
        
        # Display recommendations
        for rec in recommendations:
            if rec['priority'] == 'High':
                st.error(f"**{rec['area']}:** {rec['recommendation']}")
            elif rec['priority'] == 'Medium':
                st.warning(f"**{rec['area']}:** {rec['recommendation']}")
            else:
                st.info(f"**{rec['area']}:** {rec['recommendation']}")
    
    else:
        st.info("No analysis results available. Please run an analysis first.")

with tab4:
    st.header("Settings & Configuration")
    
    # Advanced settings
    st.subheader("Advanced Analysis Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Content Analysis")
        min_word_threshold = st.number_input("Minimum Word Count Threshold", 100, 2000, 1000)
        min_paragraph_threshold = st.number_input("Minimum Paragraph Count", 5, 20, 10)
        enable_readability = st.checkbox("Enable Readability Analysis", value=False)
        
    with col2:
        st.markdown("### Query Prediction")
        max_queries = st.slider("Max Predicted Queries per Page", 5, 20, 10)
        query_variation_types = st.multiselect(
            "Query Variation Types",
            ["What is", "How to", "Best", "Guide", "Tips", "Tutorial", "Examples"],
            default=["What is", "How to", "Guide"]
        )
    
    st.markdown("---")
    st.subheader("API Configuration")
    
    # Placeholder for additional API configurations
    st.markdown("### Additional APIs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        semrush_api = st.text_input("SEMrush API Key", type="password")
        ahrefs_api = st.text_input("Ahrefs API Key", type="password")
        
    with col2:
        openai_api = st.text_input("OpenAI API Key (for NLP)", type="password")
        gemini_api = st.text_input("Google Gemini API Key", type="password")
    
    # Save settings
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
    
    st.markdown("---")
    st.subheader("About Query Fan-Out Analysis")
    
    st.info("""
    **Query Fan-Out** is Google's advanced technique for processing search queries by:
    
    - **Expanding** single queries into multiple related sub-queries
    - **Processing** these queries in parallel across different data sources
    - **Synthesizing** results into comprehensive AI-powered responses
    
    This tool helps optimize your content for this new search paradigm by:
    - Analyzing content comprehensiveness
    - Predicting potential sub-queries
    - Identifying content gaps
    - Providing actionable optimization recommendations
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Query Fan-Out SEO Analyzer v1.0 | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
