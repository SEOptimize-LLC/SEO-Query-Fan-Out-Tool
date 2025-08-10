"""
Query Fan-Out Analysis Tool
Supports both new content planning and existing content optimization
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json

# Import modules
from config import Config
from utils import QueryAnalyzer, ContentAnalyzer, UIHelpers

# Page configuration
st.set_page_config(
    page_title="Query Fan-Out Analysis Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'mode' not in st.session_state:
    st.session_state.mode = 'new_content'

# Title and description
st.title("🔍 Query Fan-Out Analysis Tool")
st.markdown("""
Analyze and optimize content using Google's Query Fan-Out methodology to maximize 
visibility in AI-powered search results and AI Overviews.
""")

# Mode selection
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    mode = st.radio(
        "Choose your analysis mode:",
        options=['new_content', 'optimize_existing'],
        format_func=lambda x: {
            'new_content': '✍️ New Content Planning',
            'optimize_existing': '🔧 Optimize Existing Content'
        }[x],
        horizontal=True,
        help="New content mode for planning, Optimize mode for improving existing pages"
    )
    st.session_state.mode = mode

st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Gemini API Key
    st.subheader("🤖 Gemini API Configuration")
    gemini_api_key = Config.get_gemini_api_key()
    if not gemini_api_key:
        gemini_api_key = st.text_input(
            "Gemini API Key", 
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
    else:
        st.success("✅ Using Gemini API key from configuration")
        override_key = st.text_input(
            "Override API Key (optional)", 
            type="password",
            help="Enter a different API key to override the configured one"
        )
        if override_key:
            gemini_api_key = override_key
    
    # Model selection
    gemini_model = st.selectbox(
        "Gemini Model",
        options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
        index=0,
        help="Select which Gemini model to use. Flash is faster and cheaper, Pro is more capable."
    )
    
    # Analysis settings
    st.subheader("📊 Analysis Settings")
    
    # Query Fan-Out Settings based on the blog methodology
    st.subheader("🎯 Query Fan-Out Configuration")
    
    # Query Variant Types to include
    variant_types = st.multiselect(
        "Query Variant Types to Generate",
        options=[
            "equivalent",
            "follow_up",
            "generalization",
            "canonicalization",
            "entailment",
            "specification",
            "clarification"
        ],
        default=["equivalent", "follow_up", "specification", "entailment"],
        help="Select which types of query variants to generate based on Google's Fan-Out system"
    )
    
    # AI Search Type Selection
    ai_search_type = st.radio(
        "Target Optimization Type:",
        options=["ai_overviews", "ai_mode", "both"],
        format_func=lambda x: {
            'ai_overviews': '🔍 AI Overviews - Quick answers & featured snippets',
            'ai_mode': '🧠 AI Mode - Complex query fan-out & research',
            'both': '🎯 Both - Comprehensive optimization'
        }[x],
        help="Choose your optimization target"
    )
    
    analysis_depth = st.select_slider(
        "Analysis Depth",
        options=["Basic", "Standard", "Comprehensive"],
        value="Standard",
        help="How deep should the fan-out analysis go?"
    )
    
    if st.session_state.mode == 'new_content':
        max_queries = st.slider(
            "Max queries to analyze", 
            min_value=5, 
            max_value=100, 
            value=20
        )
    else:
        # For existing content, we analyze all detected topics
        max_topics = st.slider(
            "Max topics to analyze", 
            min_value=5, 
            max_value=50, 
            value=15,
            help="Maximum number of topics to extract and analyze from existing content"
        )
    
    include_schema = st.checkbox("Include Schema recommendations", value=True)
    include_competitors = st.checkbox("Include competitive analysis", value=False)
    
    # Additional options based on optimization type
    if ai_search_type in ["ai_mode", "both"]:
        include_entity_mapping = st.checkbox("Include entity relationship mapping", value=True)
        include_cross_verification = st.checkbox("Include cross-variant verification", value=True)
    else:
        include_entity_mapping = False
        include_cross_verification = False
    
    if ai_search_type in ["ai_overviews", "both"]:
        include_snippet_optimization = st.checkbox("Include snippet optimization tips", value=True)
        include_paa_optimization = st.checkbox("Include People Also Ask optimization", value=True)
    else:
        include_snippet_optimization = False
        include_paa_optimization = False

# Main content area based on mode
if st.session_state.mode == 'new_content':
    # New content planning mode
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Your Target Queries")
        
        # Add context inputs for better fan-out generation
        st.subheader("Context Information (Optional)")
        col_a, col_b = st.columns(2)
        with col_a:
            target_audience = st.text_input(
                "Target Audience",
                placeholder="e.g., SEO professionals, small business owners",
                help="Helps generate more relevant query variants"
            )
        with col_b:
            content_type = st.selectbox(
                "Content Type",
                options=["Blog Post", "Guide", "Tutorial", "Product Page", "Service Page", "Research"],
                help="Affects the types of queries generated"
            )
        
        # Text area for queries
        queries_input = st.text_area(
            "Enter queries for new content planning (one per line)",
            height=250,
            placeholder="""Example queries:
query fan out SEO
optimizing for Google AI mode
AI overviews content strategy
semantic SEO techniques
entity-based content optimization""",
            help="Enter queries you want to target with new content"
        )
        
        # Optional: Upload CSV
        uploaded_file = st.file_uploader(
            "Or upload a CSV file with queries",
            type=['csv'],
            help="CSV should have a 'query' column"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'query' in df.columns:
                    queries_input = '\n'.join(df['query'].astype(str).tolist())
                    st.success(f"✅ Loaded {len(df)} queries from CSV")
                else:
                    st.error("❌ CSV must have a 'query' column")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
    
    with col2:
        st.header("📊 Query Overview")
        queries_list = [q.strip() for q in queries_input.split('\n') if q.strip()]
        
        st.metric("Total Queries", len(queries_list))
        st.metric("Unique Queries", len(set(queries_list)))
        
        if queries_list:
            avg_length = sum(len(q.split()) for q in queries_list) / len(queries_list)
            st.metric("Avg Query Length", f"{avg_length:.1f} words")
            
            # Show query preview
            with st.expander("Query Preview"):
                for i, query in enumerate(queries_list[:10]):
                    st.write(f"{i+1}. {query}")
                if len(queries_list) > 10:
                    st.write(f"... and {len(queries_list) - 10} more")
    
    # Analysis button for new content mode
    if st.button("🚀 Run Query Fan-Out Analysis", type="primary", disabled=not gemini_api_key):
        if queries_list:
            # Prepare analysis settings
            analysis_settings = {
                'max_queries': min(max_queries, len(queries_list)),
                'depth': analysis_depth,
                'include_schema': include_schema,
                'include_competitors': include_competitors,
                'mode': 'new_content',
                'ai_search_type': ai_search_type,
                'variant_types': variant_types,
                'include_entity_mapping': include_entity_mapping,
                'include_cross_verification': include_cross_verification,
                'include_snippet_optimization': include_snippet_optimization,
                'include_paa_optimization': include_paa_optimization,
                'gemini_model': gemini_model,
                'target_audience': target_audience,
                'content_type': content_type
            }
            
            # Create DataFrame
            queries_df = pd.DataFrame({
                'query': queries_list[:max_queries],
                'priority': range(1, min(max_queries + 1, len(queries_list) + 1))
            })
            
            with st.spinner("🤖 Analyzing queries with Gemini AI..."):
                analysis = QueryAnalyzer.analyze_query_fanout_new_content(
                    queries_df,
                    gemini_api_key,
                    analysis_settings
                )
                
                if analysis:
                    # Store and display results
                    st.session_state.last_analysis = {
                        'timestamp': datetime.now(),
                        'analysis': analysis,
                        'settings': analysis_settings,
                        'queries': queries_list
                    }
                    
                    st.markdown("---")
                    st.header("📋 Query Fan-Out Analysis Results")
                    st.markdown(analysis)
                    
                    # Export options
                    UIHelpers.show_export_options(
                        analysis, 
                        queries_list, 
                        analysis_settings, 
                        mode='new_content'
                    )
        else:
            st.warning("Please enter at least one query to analyze")

elif st.session_state.mode == 'optimize_existing':
    # Existing content optimization mode
    st.header("🔧 Optimize Existing Content")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # URL input
        content_url = st.text_input(
            "Content URL",
            placeholder="https://example.com/your-content-page",
            help="Enter the URL of the content you want to optimize"
        )
        
        # Primary keyword
        primary_keyword = st.text_input(
            "Primary Target Keyword",
            placeholder="e.g., query fan out SEO",
            help="The main keyword you're targeting with this content"
        )
        
        # Optional: Additional keywords
        additional_keywords = st.text_area(
            "Additional Keywords (optional)",
            placeholder="Enter additional keywords, one per line",
            height=100,
            help="Other keywords you want to rank for"
        )
        
        # Competition URLs (optional)
        with st.expander("Competitive Analysis (Optional)"):
            competitor_urls = st.text_area(
                "Competitor URLs",
                placeholder="Enter competitor URLs, one per line",
                height=100,
                help="URLs of competing content ranking for similar keywords"
            )
    
    with col2:
        st.header("📊 Analysis Options")
        
        # Content analysis options
        analyze_readability = st.checkbox("Analyze readability", value=True)
        analyze_structure = st.checkbox("Analyze content structure", value=True)
        analyze_entities = st.checkbox("Extract and analyze entities", value=True)
        analyze_gaps = st.checkbox("Identify content gaps", value=True)
        
        # Quick tips
        with st.expander("💡 Optimization Tips"):
            st.markdown("""
            **For best results:**
            - Use your main target keyword
            - Include 3-5 related keywords
            - Add 2-3 competitor URLs if available
            - Ensure the URL is publicly accessible
            """)
    
    # Analysis button for existing content
    if st.button("🔍 Analyze & Optimize Content", type="primary", disabled=not gemini_api_key):
        if content_url and primary_keyword:
            with st.spinner("📥 Fetching and analyzing content..."):
                # Fetch the content
                content_data = ContentAnalyzer.fetch_content(content_url)
                
                if content_data:
                    # Extract additional keywords list
                    additional_kw_list = [kw.strip() for kw in additional_keywords.split('\n') if kw.strip()]
                    competitor_url_list = [url.strip() for url in competitor_urls.split('\n') if url.strip()]
                    
                    # Prepare analysis settings
                    analysis_settings = {
                        'depth': analysis_depth,
                        'include_schema': include_schema,
                        'include_competitors': include_competitors and len(competitor_url_list) > 0,
                        'mode': 'optimize_existing',
                        'ai_search_type': ai_search_type,
                        'variant_types': variant_types,
                        'include_entity_mapping': include_entity_mapping,
                        'include_cross_verification': include_cross_verification,
                        'include_snippet_optimization': include_snippet_optimization,
                        'include_paa_optimization': include_paa_optimization,
                        'gemini_model': gemini_model,
                        'analyze_readability': analyze_readability,
                        'analyze_structure': analyze_structure,
                        'analyze_entities': analyze_entities,
                        'analyze_gaps': analyze_gaps,
                        'max_topics': max_topics
                    }
                    
                    # Show content overview
                    st.markdown("---")
                    st.subheader("📄 Content Overview")
                    UIHelpers.display_content_metrics(content_data)
                    
                    # Run the analysis
                    with st.spinner("🤖 Running Query Fan-Out analysis on your content..."):
                        analysis = ContentAnalyzer.analyze_existing_content(
                            content_data,
                            primary_keyword,
                            additional_kw_list,
                            competitor_url_list,
                            gemini_api_key,
                            analysis_settings
                        )
                        
                        if analysis:
                            # Store and display results
                            st.session_state.last_analysis = {
                                'timestamp': datetime.now(),
                                'analysis': analysis,
                                'settings': analysis_settings,
                                'url': content_url,
                                'primary_keyword': primary_keyword
                            }
                            
                            st.markdown("---")
                            st.header("📋 Content Optimization Analysis")
                            st.markdown(analysis)
                            
                            # Export options
                            UIHelpers.show_export_options(
                                analysis,
                                {'url': content_url, 'keyword': primary_keyword},
                                analysis_settings,
                                mode='optimize_existing'
                            )
                else:
                    st.error("❌ Could not fetch content from the provided URL. Please check the URL and try again.")
        else:
            st.warning("Please provide both a URL and primary keyword to analyze")

# Footer
st.markdown("---")
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style='text-align: center; color: #666;'>
                <p>Query Fan-Out Analysis Tool | Built with ❤️</p>
                <p>
                    <a href='https://dejan.ai/blog/googles-query-fan-out-system-a-technical-overview/' target='_blank'>Learn about Query Fan-Out</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
