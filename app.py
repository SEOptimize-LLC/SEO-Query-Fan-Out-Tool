"""
Query Fan-Out Analysis Tool
Supports both new content planning and existing content optimization
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json

# Import modules with error handling
try:
    from config import Config
    from utils import QueryAnalyzer, ContentAnalyzer, UIHelpers, APIValidator
except ImportError as e:
    st.error(f"Import Error: {str(e)}")
    st.error("Please ensure config.py and utils.py are in the same directory")
    st.stop()

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
if 'api_keys' not in st.session_state:
    # Try to load from Config
    try:
        st.session_state.api_keys = Config.get_all_api_keys()
    except:
        st.session_state.api_keys = {
            'gemini': '',
            'openai': '',
            'anthropic': ''
        }

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
    
    # API Keys Section
    st.subheader("🔑 API Keys Configuration")
    
    with st.expander("API Keys Setup", expanded=True):
        st.markdown("Enter API keys for the providers you want to use:")
        
        # Gemini API Key
        st.markdown("**Google Gemini**")
        gemini_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.api_keys.get('gemini', ''),
            type="password",
            help="Get from: https://makersuite.google.com/app/apikey",
            key="gemini_key_input"
        )
        if gemini_key != st.session_state.api_keys.get('gemini', ''):
            st.session_state.api_keys['gemini'] = gemini_key
            
        if gemini_key and st.button("Validate", key="validate_gemini"):
            with st.spinner("Validating..."):
                is_valid, message = APIValidator.validate_gemini_key(gemini_key)
                if is_valid:
                    st.success(message)
                else:
                    st.error(message)
        
        st.markdown("---")
        
        # OpenAI API Key
        st.markdown("**OpenAI**")
        openai_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.api_keys.get('openai', ''),
            type="password",
            help="Get from: https://platform.openai.com/api-keys",
            key="openai_key_input"
        )
        if openai_key != st.session_state.api_keys.get('openai', ''):
            st.session_state.api_keys['openai'] = openai_key
            
        if openai_key and st.button("Validate", key="validate_openai"):
            with st.spinner("Validating..."):
                is_valid, message = APIValidator.validate_openai_key(openai_key)
                if is_valid:
                    st.success(message)
                else:
                    st.error(message)
        
        st.markdown("---")
        
        # Anthropic API Key
        st.markdown("**Anthropic Claude**")
        anthropic_key = st.text_input(
            "Anthropic API Key",
            value=st.session_state.api_keys.get('anthropic', ''),
            type="password",
            help="Get from: https://console.anthropic.com/settings/keys",
            key="anthropic_key_input"
        )
        if anthropic_key != st.session_state.api_keys.get('anthropic', ''):
            st.session_state.api_keys['anthropic'] = anthropic_key
            
        if anthropic_key and st.button("Validate", key="validate_anthropic"):
            with st.spinner("Validating..."):
                is_valid, message = APIValidator.validate_anthropic_key(anthropic_key)
                if is_valid:
                    st.success(message)
                else:
                    st.error(message)
    
    # Show which providers are configured
    configured_providers = [p for p, k in st.session_state.api_keys.items() if k]
    if configured_providers:
        st.success(f"✅ Configured: {', '.join(configured_providers)}")
    else:
        st.warning("⚠️ No API keys configured yet")
    
    st.markdown("---")
    
    # Model Selection
    st.subheader("🤖 AI Model Selection")
    
    # Only show providers that have API keys configured
    available_providers = [p for p, k in st.session_state.api_keys.items() if k]
    
    if not available_providers:
        st.warning("Please configure at least one API key above")
        ai_provider = None
        selected_model = None
    else:
        ai_provider = st.selectbox(
            "Select AI Provider",
            options=available_providers,
            format_func=lambda x: Config.AI_MODELS[x]['name'],
            help="Choose from your configured providers"
        )
        
        # Model selection based on provider
        selected_model_info = st.selectbox(
            "Select Model",
            options=Config.AI_MODELS[ai_provider]['models'],
            format_func=lambda x: x['name'],
            help="Choose the specific model to use"
        )
        
        selected_model = selected_model_info['id']
        st.caption(f"ℹ️ {selected_model_info['description']}")
        
        # Show model-specific tips
        if ai_provider == 'openai' and 'o1' in selected_model:
            st.info("💡 O1 models excel at complex reasoning and multi-step analysis")
        elif ai_provider == 'anthropic' and 'opus' in selected_model:
            st.info("💡 Claude Opus provides the most comprehensive analysis")
        elif ai_provider == 'gemini' and 'pro' in selected_model:
            st.info("💡 Gemini Pro offers excellent balance of speed and quality")
    
    st.markdown("---")
    
    # Analysis settings
    st.subheader("📊 Analysis Settings")
    
    # Query Fan-Out Configuration
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
    button_disabled = not available_providers or not queries_list
    
    if st.button("🚀 Run Query Fan-Out Analysis", type="primary", disabled=button_disabled):
        if queries_list and ai_provider:
            # Get the API key for selected provider
            api_key = st.session_state.api_keys.get(ai_provider)
            
            if not api_key:
                st.error(f"No API key configured for {Config.AI_MODELS[ai_provider]['name']}")
            else:
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
                    'ai_provider': ai_provider,
                    'model': selected_model,
                    'target_audience': target_audience,
                    'content_type': content_type
                }
                
                # Create DataFrame
                queries_df = pd.DataFrame({
                    'query': queries_list[:max_queries],
                    'priority': range(1, min(max_queries + 1, len(queries_list) + 1))
                })
                
                with st.spinner(f"🤖 Analyzing queries with {Config.AI_MODELS[ai_provider]['name']} ({selected_model})..."):
                    try:
                        analysis = QueryAnalyzer.analyze_query_fanout_new_content(
                            queries_df,
                            api_key,
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
                            st.error("Analysis failed. Please check your API key and try again.")
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        st.info("Please check your API key and model selection")
        elif not available_providers:
            st.warning("Please configure at least one API key in the sidebar")
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
    button_disabled = not available_providers or not (content_url and primary_keyword)
    
    if st.button("🔍 Analyze & Optimize Content", type="primary", disabled=button_disabled):
        if content_url and primary_keyword and ai_provider:
            # Get the API key for selected provider
            api_key = st.session_state.api_keys.get(ai_provider)
            
            if not api_key:
                st.error(f"No API key configured for {Config.AI_MODELS[ai_provider]['name']}")
            else:
                with st.spinner("📥 Fetching and analyzing content..."):
                    try:
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
                                'ai_provider': ai_provider,
                                'model': selected_model,
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
                            with st.spinner(f"🤖 Running Query Fan-Out analysis with {Config.AI_MODELS[ai_provider]['name']} ({selected_model})..."):
                                analysis = ContentAnalyzer.analyze_existing_content(
                                    content_data,
                                    primary_keyword,
                                    additional_kw_list,
                                    competitor_url_list,
                                    api_key,
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
                                    st.error("Analysis failed. Please check your API key and try again.")
                        else:
                            st.error("❌ Could not fetch content from the provided URL. Please check the URL and try again.")
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        st.info("Please ensure the URL is accessible and try again")
        elif not available_providers:
            st.warning("Please configure at least one API key in the sidebar")
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

# Debug info (only show in development)
if st.checkbox("Show Debug Info", value=False, key="debug"):
    st.markdown("### Debug Information")
    st.json({
        "configured_providers": configured_providers if 'configured_providers' in locals() else [],
        "selected_provider": ai_provider if 'ai_provider' in locals() else None,
        "selected_model": selected_model if 'selected_model' in locals() else None,
        "mode": st.session_state.mode,
        "api_keys_configured": {k: bool(v) for k, v in st.session_state.api_keys.items()}
    })
