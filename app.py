"""
Query Fan-Out Analysis Tool
Main Streamlit Application
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

# Title and description
st.title("🔍 Query Fan-Out Analysis Tool")
st.markdown("""
Analyze your Google Search Console data using AI-powered Query Fan-Out methodology to optimize 
content for Google's AI Mode search.
""")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # OAuth status
    if st.session_state.get('authenticated', False):
        st.success("✅ Connected to Google Search Console")
        if st.button("🚪 Logout", key="logout_button"):
            GSCAuth.logout()
            st.rerun()
    else:
        st.info("📌 Connect to Google Search Console to begin")
    
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
        # Allow override
        override_key = st.text_input(
            "Override API Key (optional)", 
            type="password",
            help="Enter a different API key to override the configured one"
        )
        if override_key:
            gemini_api_key = override_key
    
    # Analysis parameters
    st.subheader("📊 Analysis Parameters")
    days_back = st.slider(
        "Days of data to analyze", 
        min_value=7, 
        max_value=90, 
        value=Config.DEFAULT_DAYS_BACK
    )
    
    # Sorting preference
    st.subheader("🔄 Data Sorting Preference")
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
    
    # Query Fan-Out settings
    st.subheader("🎯 Query Fan-Out Settings")
    max_queries = st.slider(
        "Max queries to analyze", 
        min_value=10, 
        max_value=100, 
        value=Config.DEFAULT_MAX_QUERIES
    )
    
    # Brand terms for filtering
    brand_terms_input = st.text_area(
        "Brand terms to exclude (one per line)",
        help="Enter brand-related terms to filter out branded queries"
    )
    brand_terms = [term.strip() for term in brand_terms_input.split('\n') if term.strip()]
    
    include_branded = st.checkbox("Include branded queries", value=False)
    include_schema = st.checkbox("Include Schema recommendations", value=True)
    include_competitors = st.checkbox("Include competitive analysis", value=False)

# Main content area
if not st.session_state.get('authenticated', False):
    # Authentication required
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔐 Connect to Google Search Console")
        st.markdown("""
        To analyze your search queries, you need to connect your Google Search Console account.
        
        **Benefits of connecting GSC:**
        - Analyze real queries that drive traffic to your site
        - See performance metrics (clicks, impressions, CTR, position)
        - Identify optimization opportunities based on actual data
        - Prioritize content improvements with data-driven insights
        """)
        
        if GSCAuth.authenticate():
            st.rerun()
    
    with col2:
        st.info("""
        **📋 Requirements:**
        - Google Search Console access
        - Verified website property
        - OAuth credentials configured
        
        **🔒 Privacy:**
        - Data stays in your session
        - No data is stored permanently
        - You can logout anytime
        """)

else:
    # Authenticated - Show main interface
    
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
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("📊 Fetch GSC Data", type="primary", use_container_width=True):
            with st.spinner(f"Fetching data for the last {days_back} days..."):
                df = GSCData.fetch_query_data(selected_site, days_back)
                
                if df is not None and not df.empty:
                    # Apply filters
                    df_filtered = QueryAnalyzer.filter_queries(
                        df, 
                        min_value, 
                        filter_column,
                        include_branded,
                        brand_terms
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
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Query Data", 
            "📊 Visualizations", 
            "🤖 AI Analysis", 
            "📥 Export"
        ])
        
        with tab1:
            # Query data table
            st.subheader(f"Top Queries by {sort_metric.upper()}")
            
            # Sort data appropriately
            display_df = st.session_state.gsc_data.copy()
            if sort_metric == 'position':
                display_df = display_df.sort_values('position')
            else:
                display_df = display_df.sort_values(sort_metric, ascending=False)
            
            # Format for display
            display_df['ctr'] = display_df['ctr'].apply(lambda x: f"{x:.2%}")
            display_df['position'] = display_df['position'].round(1)
            
            st.dataframe(
                display_df.head(50),
                use_container_width=True,
                hide_index=True
            )
        
        with tab2:
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Query Distribution")
                if sort_metric == 'position':
                    chart_data = st.session_state.gsc_data.nsmallest(15, sort_metric)
                else:
                    chart_data = st.session_state.gsc_data.nlargest(15, sort_metric)
                
                st.bar_chart(
                    chart_data.set_index('query')[sort_metric],
                    height=400
                )
            
            with col2:
                st.subheader("📍 Position Distribution")
                position_dist = UIHelpers.create_position_distribution(st.session_state.gsc_data)
                st.bar_chart(position_dist, height=400)
        
        with tab3:
            # AI Analysis
            st.subheader("🤖 Query Fan-Out AI Analysis")
            
            if not gemini_api_key:
                st.warning("⚠️ Please provide a Gemini API key in the sidebar to enable AI analysis")
            else:
                # Analysis settings
                analysis_settings = {
                    'max_queries': max_queries,
                    'sort_metric': sort_metric,
                    'include_schema': include_schema,
                    'include_competitors': include_competitors,
                    'depth': 'comprehensive'
                }
                
                if st.button("🚀 Run Query Fan-Out Analysis", type="primary", use_container_width=True):
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
                            
                            # Add to history
                            st.session_state.analysis_history.append(st.session_state.last_analysis)
                            
                            # Display analysis
                            st.markdown(analysis)
                
                # Display last analysis if available
                elif 'last_analysis' in st.session_state:
                    st.info(f"Showing analysis from {st.session_state.last_analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.markdown(st.session_state.last_analysis['analysis'])
        
        with tab4:
            # Export options
            st.subheader("📥 Export Data & Analysis")
            
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
                if 'last_analysis' in st.session_state:
                    report = QueryAnalyzer.export_analysis(
                        st.session_state.last_analysis['analysis'],
                        st.session_state.gsc_data,
                        format='markdown'
                    )
                    st.download_button(
                        label="📄 Download Analysis Report",
                        data=report,
                        file_name=f"query_fanout_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                else:
                    st.info("Run AI analysis first to download report")
            
            with col3:
                # Export JSON for automation
                if 'last_analysis' in st.session_state:
                    export_data = {
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
                        'top_queries': st.session_state.gsc_data.head(20).to_dict('records'),
                        'analysis': st.session_state.last_analysis['analysis']
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
