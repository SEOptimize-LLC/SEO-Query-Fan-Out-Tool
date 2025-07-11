"""
Query Fan-Out Analysis Tool - Dual Mode
Supports both GSC integration and manual query input
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json

# Import modules
from config import Config
from gsc_integration import GSCAuth, GSCData
from utils import QueryAnalyzer, UIHelpers

# Page configuration
st.set_page_config(
    page_title="Query Fan-Out Analysis Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'mode' not in st.session_state:
    st.session_state.mode = 'manual'  # Default to manual mode

# Title and description
st.title("🔍 Query Fan-Out Analysis Tool")
st.markdown("""
Analyze search queries using AI-powered Query Fan-Out methodology to optimize 
content for Google's AI Mode search and AI Overviews.
""")

# Mode selection
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    mode = st.radio(
        "Choose your analysis mode:",
        options=['manual', 'gsc'],
        format_func=lambda x: {
            'manual': '✍️ Manual Query Input (New Content Planning)',
            'gsc': '📊 Google Search Console (Optimize Existing Content)'
        }[x],
        horizontal=True,
        help="Manual mode for planning new content, GSC mode for optimizing existing pages"
    )
    st.session_state.mode = mode

st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Mode-specific configuration
    if st.session_state.mode == 'gsc':
        # OAuth status
        if st.session_state.get('authenticated', False):
            st.success("✅ Connected to Google Search Console")
            if st.button("🚪 Logout", key="logout_button"):
                GSCAuth.logout()
                st.rerun()
        else:
            st.info("📌 Connect to Google Search Console to begin")
    
    # Gemini API Key (needed for both modes)
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
        # Allow override
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
    
    if st.session_state.mode == 'gsc':
        # GSC-specific settings
        days_back = st.slider(
            "Days of data to analyze", 
            min_value=7, 
            max_value=90, 
            value=Config.DEFAULT_DAYS_BACK
        )
        
        # Sorting preference
        sort_metric = st.radio(
            "Sort queries by:",
            ["clicks", "impressions", "ctr", "position"],
            index=0,
            help="Choose which metric to prioritize"
        )
        
        # Dynamic filter
        if sort_metric == "clicks":
            min_value = st.number_input(
                "Minimum clicks", 
                min_value=1, 
                max_value=100, 
                value=Config.DEFAULT_MIN_CLICKS
            )
            filter_column = "clicks"
        elif sort_metric == "impressions":
            min_value = st.number_input(
                "Minimum impressions", 
                min_value=10, 
                max_value=1000, 
                value=50
            )
            filter_column = "impressions"
        elif sort_metric == "ctr":
            min_value = st.slider(
                "Minimum CTR (%)", 
                min_value=0.0, 
                max_value=10.0, 
                value=0.5, 
                step=0.1
            )
            filter_column = "ctr"
            min_value = min_value / 100
        else:  # position
            min_value = st.slider(
                "Maximum position", 
                min_value=1, 
                max_value=100, 
                value=50
            )
            filter_column = "position"
    
    # Common settings for both modes
    st.subheader("🎯 Query Fan-Out Settings")
    
    # AI Search Type Selection
    ai_search_type = st.radio(
        "Target AI Search Type:",
        options=["ai_overviews", "ai_mode"],
        format_func=lambda x: {
            'ai_overviews': '🔍 AI Overviews (Simple) - Quick answers & featured snippets',
            'ai_mode': '🧠 AI Mode (Complex) - Full query fan-out & multi-step research'
        }[x],
        help="""
        **AI Overviews**: Optimizes for quick, direct answers that appear at the top of search results
        **AI Mode**: Optimizes for complex query expansions and multi-step research journeys
        """
    )
    
    analysis_depth = st.select_slider(
        "Analysis Depth",
        options=["Basic", "Standard", "Comprehensive"],
        value="Standard",
        help="How deep should the fan-out analysis go?"
    )
    
    max_queries = st.slider(
        "Max queries to analyze", 
        min_value=5, 
        max_value=100, 
        value=20
    )
    
    include_schema = st.checkbox("Include Schema recommendations", value=True)
    include_competitors = st.checkbox("Include competitive analysis", value=False)
    
    # Additional options based on AI search type
    if ai_search_type == "ai_mode":
        include_followup = st.checkbox("Include follow-up query predictions", value=True)
        include_entity_mapping = st.checkbox("Include entity relationship mapping", value=True)
        include_snippet_optimization = False
        include_paa_optimization = False
    else:
        include_snippet_optimization = st.checkbox("Include snippet optimization tips", value=True)
        include_paa_optimization = st.checkbox("Include People Also Ask optimization", value=True)
        include_followup = False
        include_entity_mapping = False
    
    if st.session_state.mode == 'gsc':
        # Brand filtering for GSC mode
        brand_terms_input = st.text_area(
            "Brand terms to exclude (one per line)",
            help="Enter brand-related terms to filter out branded queries"
        )
        brand_terms = [term.strip() for term in brand_terms_input.split('\n') if term.strip()]
        include_branded = st.checkbox("Include branded queries", value=False)
    else:
        # Initialize these for manual mode
        brand_terms = []
        include_branded = True

# Main content area based on mode
if st.session_state.mode == 'manual':
    # Manual query input mode
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Your Target Queries")
        
        # Text area for queries
        queries_input = st.text_area(
            "Enter queries for new content planning (one per line)",
            height=250,
            placeholder="""Example queries for a new AI SEO guide:
how does Google AI mode work
query fan out SEO strategy
optimizing content for AI search
Google Gemini search optimization
AI overviews content strategy
passage ranking optimization
semantic SEO for AI""",
            help="Enter queries you want to target with new content. These should be queries you've researched but haven't created content for yet."
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
    
    # Analysis button for manual mode
    if st.button("🚀 Run Query Fan-Out Analysis", type="primary", disabled=not gemini_api_key):
        if queries_list:
            # Prepare analysis settings
            analysis_settings = {
                'max_queries': min(max_queries, len(queries_list)),
                'depth': analysis_depth,
                'include_schema': include_schema,
                'include_competitors': include_competitors,
                'mode': 'manual',
                'ai_search_type': ai_search_type,
                'include_followup': include_followup,
                'include_entity_mapping': include_entity_mapping,
                'include_snippet_optimization': include_snippet_optimization,
                'include_paa_optimization': include_paa_optimization,
                'gemini_model': gemini_model
            }
            
            # Create a simple DataFrame for consistency
            queries_df = pd.DataFrame({
                'query': queries_list[:max_queries],
                'priority': range(1, min(max_queries + 1, len(queries_list) + 1))
            })
            
            with st.spinner("🤖 Analyzing queries with Gemini AI..."):
                analysis = QueryAnalyzer.analyze_query_fanout_manual(
                    queries_df,
                    gemini_api_key,
                    analysis_settings
                )
                
                if analysis:
                    # Store results
                    st.session_state.last_analysis = {
                        'timestamp': datetime.now(),
                        'analysis': analysis,
                        'settings': analysis_settings,
                        'queries': queries_list
                    }
                    
                    # Display results
                    st.markdown("---")
                    st.header("📋 Query Fan-Out Analysis Results")
                    st.markdown(analysis)
                    
                    # Export options
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            label="📥 Download Analysis (Markdown)",
                            data=analysis,
                            file_name=f"query_fanout_new_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    
                    with col2:
                        # Create a detailed report
                        report = f"""# Query Fan-Out Analysis Report - New Content Planning
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Queries Analyzed ({len(queries_list[:max_queries])} queries)
{chr(10).join(f"- {q}" for q in queries_list[:max_queries])}

## Analysis Settings
- Depth: {analysis_depth}
- Schema Recommendations: {'Yes' if include_schema else 'No'}
- Competitive Analysis: {'Yes' if include_competitors else 'No'}

## Analysis Results
{analysis}

---
*Report generated by Query Fan-Out Analysis Tool - Manual Mode*
"""
                        st.download_button(
                            label="📄 Download Full Report",
                            data=report,
                            file_name=f"query_fanout_report_new_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    
                    with col3:
                        # JSON export
                        json_data = {
                            'mode': 'manual',
                            'timestamp': datetime.now().isoformat(),
                            'queries': queries_list[:max_queries],
                            'settings': analysis_settings,
                            'analysis': analysis
                        }
                        st.download_button(
                            label="💾 Download JSON",
                            data=json.dumps(json_data, indent=2),
                            file_name=f"query_fanout_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
        else:
            st.warning("Please enter at least one query to analyze")

elif st.session_state.mode == 'gsc':
    # GSC integration mode
    if not st.session_state.get('authenticated', False):
        # Authentication required
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🔐 Connect to Google Search Console")
            st.markdown("""
            To optimize existing content, connect your Google Search Console account to analyze real search data.
            
            **Benefits of GSC integration:**
            - Analyze queries that already drive traffic
            - See actual performance metrics (CTR, position)
            - Identify quick-win optimization opportunities
            - Prioritize based on real data, not assumptions
            """)
            
            if GSCAuth.authenticate():
                st.rerun()
        
        with col2:
            st.info("""
            **📋 Use cases:**
            - Optimize underperforming pages
            - Find content gaps in existing topics
            - Improve rankings for valuable queries
            - Enhance meta descriptions for better CTR
            
            **🔒 Privacy:**
            - Data stays in your session
            - No permanent storage
            - Logout anytime
            """)
    
    else:
        # Authenticated - Show GSC interface
        # Load sites if not already loaded
        if 'sites' not in st.session_state:
            with st.spinner("Loading Search Console properties..."):
                st.session_state.sites = GSCData.get_sites()
        
        if not st.session_state.sites:
            st.warning("No verified sites found in your Search Console account.")
            st.stop()
        
        # Site selection
        selected_site = st.selectbox(
            "Select a Search Console property:",
            st.session_state.sites,
            help="Choose the website you want to analyze"
        )
        
        # Data fetching section
        if st.button("📊 Fetch GSC Data", type="primary", use_container_width=True):
            with st.spinner(f"Fetching data for the last {days_back} days..."):
                df = GSCData.fetch_query_data(selected_site, days_back)
                
                if df is not None and not df.empty:
                    # Apply filters
                    df_filtered = QueryAnalyzer.filter_queries(
                        df, 
                        min_value, 
                        filter_column,
                        include_branded if 'include_branded' in locals() else True,
                        brand_terms if 'brand_terms' in locals() else []
                    )
                    
                    st.session_state.gsc_data = df_filtered
                    st.session_state.last_fetch = datetime.now()
                    st.success(f"✅ Loaded {len(df_filtered)} queries ({len(df)} before filtering)")
                else:
                    st.error("No data retrieved. Please check your date range and try again.")
        
        # Display data and analysis
        if 'gsc_data' in st.session_state and not st.session_state.gsc_data.empty:
            st.markdown("---")
            
            # Summary metrics
            st.header("📈 Performance Overview")
            UIHelpers.display_metrics(st.session_state.gsc_data)
            
            # Opportunities section
            opportunities = UIHelpers.highlight_opportunities(st.session_state.gsc_data)
            if opportunities:
                with st.expander("💡 Quick Optimization Opportunities", expanded=True):
                    for opp in opportunities:
                        st.subheader(opp['type'])
                        st.write(f"**Action:** {opp['action']}")
                        st.write("**Example queries:**", ', '.join(opp['queries'][:3]))
                        st.markdown("---")
            
            # Analysis section
            if st.button("🤖 Run Query Fan-Out Analysis", type="primary", use_container_width=True):
                if not gemini_api_key:
                    st.warning("⚠️ Please provide a Gemini API key in the sidebar")
                else:
                    # Analysis settings
                    analysis_settings = {
                        'max_queries': max_queries,
                        'sort_metric': sort_metric,
                        'include_schema': include_schema,
                        'include_competitors': include_competitors,
                        'depth': analysis_depth,
                        'mode': 'gsc',
                        'ai_search_type': ai_search_type,
                        'include_followup': include_followup,
                        'include_entity_mapping': include_entity_mapping,
                        'include_snippet_optimization': include_snippet_optimization,
                        'include_paa_optimization': include_paa_optimization,
                        'gemini_model': gemini_model
                    }
                    
                    with st.spinner("Analyzing queries with Gemini AI..."):
                        analysis = QueryAnalyzer.analyze_query_fanout(
                            st.session_state.gsc_data,
                            gemini_api_key,
                            analysis_settings
                        )
                        
                        if analysis:
                            # Store in session state
                            st.session_state.last_analysis = {
                                'timestamp': datetime.now(),
                                'analysis': analysis,
                                'settings': analysis_settings,
                                'site': selected_site
                            }
                            
                            # Display analysis
                            st.markdown("---")
                            st.header("📋 Query Fan-Out Analysis Results")
                            st.markdown(analysis)
                            
                            # Export options
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # Export raw data
                                csv = st.session_state.gsc_data.to_csv(index=False)
                                st.download_button(
                                    label="📊 Download Query Data (CSV)",
                                    data=csv,
                                    file_name=f"gsc_queries_{selected_site.replace('https://', '').replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv"
                                )
                            
                            with col2:
                                # Export analysis report
                                report = QueryAnalyzer.export_analysis(
                                    analysis,
                                    st.session_state.gsc_data,
                                    format='markdown'
                                )
                                st.download_button(
                                    label="📄 Download Analysis Report",
                                    data=report,
                                    file_name=f"query_fanout_report_gsc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                    mime="text/markdown"
                                )
                            
                            with col3:
                                # Export JSON
                                export_data = {
                                    'mode': 'gsc',
                                    'site': selected_site,
                                    'analysis_date': datetime.now().isoformat(),
                                    'settings': analysis_settings,
                                    'metrics': {
                                        'total_queries': len(st.session_state.gsc_data),
                                        'total_clicks': int(st.session_state.gsc_data['clicks'].sum()),
                                        'total_impressions': int(st.session_state.gsc_data['impressions'].sum()),
                                        'avg_ctr': float(st.session_state.gsc_data['ctr'].mean()),
                                        'avg_position': float(st.session_state.gsc_data['position'].mean())
                                    },
                                    'analysis': analysis
                                }
                                
                                st.download_button(
                                    label="💾 Download JSON Export",
                                    data=json.dumps(export_data, indent=2),
                                    file_name=f"query_fanout_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )

# Footer
st.markdown("---")
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style='text-align: center; color: #666;'>
                <p>Query Fan-Out Analysis Tool | Built with ❤️ by SEOptimize LLC</p>
                <p>
                    <a href='https://github.com/SEOptimize-LLC/SEO-Query-Fan-Out-Tool' target='_blank'>GitHub</a> | 
                    <a href='https://searchengineland.com/google-ai-mode-query-fan-out-seo-449474' target='_blank'>Learn about Query Fan-Out</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
