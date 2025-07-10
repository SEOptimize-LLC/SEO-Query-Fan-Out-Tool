import streamlit as st
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import json
import os
from datetime import datetime, timedelta
import google.generativeai as genai
import base64

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'credentials' not in st.session_state:
    st.session_state.credentials = None

# OAuth 2.0 configuration
SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']

# Streamlit app configuration
st.set_page_config(
    page_title="SEO Query Fan-Out Analysis Tool",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 SEO Query Fan-Out Analysis Tool")
st.markdown("Analyze your Google Search Console queries and get AI-powered content recommendations using Query Fan-Out methodology")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # OAuth 2.0 Client Configuration
    st.subheader("Google OAuth 2.0 Setup")
    
    with st.expander("ℹ️ How to get OAuth 2.0 credentials"):
        st.markdown("""
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project or select existing
        3. Enable Google Search Console API
        4. Create OAuth 2.0 credentials
        5. For deployment on Streamlit Cloud:
           - Add `https://[your-app-name].streamlit.app` as redirect URI
           - For local testing add `http://localhost:8501`
        6. Download the credentials JSON file
        """)
    
    # Check for credentials in Streamlit secrets first
    use_secrets = False
    try:
        if 'GOOGLE_CLIENT_ID' in st.secrets and 'GOOGLE_CLIENT_SECRET' in st.secrets:
            st.session_state.client_id = st.secrets['GOOGLE_CLIENT_ID']
            st.session_state.client_secret = st.secrets['GOOGLE_CLIENT_SECRET']
            use_secrets = True
            st.success("✅ Using credentials from Streamlit secrets")
    except:
        pass
    
    if not use_secrets:
        # Option to upload credentials file
        uploaded_file = st.file_uploader("Upload OAuth 2.0 credentials JSON", type=['json'])
        
        if uploaded_file is not None:
            credentials_info = json.load(uploaded_file)
            st.session_state.client_id = credentials_info['web']['client_id']
            st.session_state.client_secret = credentials_info['web']['client_secret']
            st.success("✅ Credentials loaded from file")
    
    # Gemini API Key
    st.subheader("Gemini API Configuration")
    try:
        if 'GEMINI_API_KEY' in st.secrets:
            gemini_api_key = st.secrets['GEMINI_API_KEY']
            st.success("✅ Using Gemini API key from secrets")
        else:
            gemini_api_key = st.text_input("Gemini API Key", type="password")
    except:
        gemini_api_key = st.text_input("Gemini API Key", type="password")
    
    # Analysis parameters
    st.subheader("Analysis Parameters")
    days_back = st.slider("Days of data to analyze", 7, 90, 30)
    
    # Data Sorting Preference (moved up)
    st.subheader("Data Sorting Preference")
    sort_metric = st.radio(
        "Sort queries by:",
        ["clicks", "impressions", "ctr", "position"],
        index=0,  # Default to clicks
        help="Choose which metric to prioritize when selecting top queries"
    )
    
    # Dynamic filter based on selected metric
    if sort_metric == "clicks":
        min_value = st.number_input("Minimum clicks", 1, 100, 5)
        filter_column = "clicks"
    elif sort_metric == "impressions":
        min_value = st.number_input("Minimum impressions", 10, 1000, 50)
        filter_column = "impressions"
    elif sort_metric == "ctr":
        min_value = st.slider("Minimum CTR (%)", 0.0, 10.0, 0.5, step=0.1)
        filter_column = "ctr"
        min_value = min_value / 100  # Convert percentage to decimal
    else:  # position
        min_value = st.slider("Maximum position", 1, 100, 50)
        filter_column = "position"
    
    # Query Fan-Out settings
    st.subheader("Query Fan-Out Settings")
    max_queries = st.slider("Max queries to analyze", 10, 100, 20)
    include_branded = st.checkbox("Include branded queries", value=False)

# Function to get redirect URI
def get_redirect_uri():
    """Get the appropriate redirect URI based on environment"""
    # Check if running on Streamlit Cloud
    # Streamlit Cloud sets specific environment variables
    if any(key in os.environ for key in ['STREAMLIT_SHARING_MODE', 'STREAMLIT_SERVER_HEADLESS']):
        # Running on Streamlit Cloud
        try:
            # First try to get from secrets
            return st.secrets.get('REDIRECT_URI', 'https://seo-query-fan-out-tool.streamlit.app')
        except:
            # Fallback to your app URL
            return 'https://seo-query-fan-out-tool.streamlit.app'  # Update this to your actual app URL
    else:
        # Local development
        return 'http://localhost:8501'

# OAuth 2.0 Authentication Function
def authenticate_google():
    """Handle Google OAuth 2.0 authentication flow"""
    
    if not hasattr(st.session_state, 'client_id') or not hasattr(st.session_state, 'client_secret'):
        st.error("Please provide OAuth 2.0 credentials (upload file or add to Streamlit secrets)")
        return False
    
    redirect_uri = get_redirect_uri()
    
    # Debug info (remove in production)
    with st.expander("🔧 Debug Info", expanded=False):
        st.write(f"Redirect URI being used: `{redirect_uri}`")
        st.write(f"Running on Streamlit Cloud: {any(key in os.environ for key in ['STREAMLIT_SHARING_MODE', 'STREAMLIT_SERVER_HEADLESS'])}")
    
    # Create flow instance
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": st.session_state.client_id,
                "client_secret": st.session_state.client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [redirect_uri]
            }
        },
        scopes=SCOPES
    )
    
    flow.redirect_uri = redirect_uri
    
    # Check if we have authorization code in URL
    query_params = st.query_params
    auth_code = query_params.get("code", None)
    
    if auth_code and not st.session_state.authenticated:
        # Exchange authorization code for credentials
        try:
            flow.fetch_token(code=auth_code)
            st.session_state.credentials = flow.credentials
            st.session_state.authenticated = True
            # Clear the URL parameters
            st.query_params.clear()
            st.success("✅ Successfully authenticated with Google!")
            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")
            return False
    
    elif not st.session_state.authenticated:
        # Generate authorization URL with prompt to select account
        auth_url, _ = flow.authorization_url(
            prompt="select_account",
            access_type="offline",
            include_granted_scopes="true"
        )
        
        st.markdown("### 🔐 Google Authentication Required")
        st.markdown("Click the button below to authenticate with your Google account:")
        st.markdown(f"[🔗 Authenticate with Google]({auth_url})")
        st.info("After authentication, you'll be redirected back to this app")
        return False
    
    return True

# Function to get GSC data
def get_gsc_data(site_url, days_back):
    """Fetch data from Google Search Console"""
    
    if not st.session_state.credentials:
        st.error("Not authenticated")
        return None
    
    try:
        # Build the service
        service = build('searchconsole', 'v1', credentials=st.session_state.credentials)
        
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        # Request data
        request = {
            'startDate': start_date.strftime('%Y-%m-%d'),
            'endDate': end_date.strftime('%Y-%m-%d'),
            'dimensions': ['query'],
            'rowLimit': 1000,
            'dataState': 'all'
        }
        
        response = service.searchanalytics().query(
            siteUrl=site_url,
            body=request
        ).execute()
        
        # Convert to DataFrame
        if 'rows' in response:
            data = []
            for row in response['rows']:
                data.append({
                    'query': row['keys'][0],
                    'clicks': row['clicks'],
                    'impressions': row['impressions'],
                    'ctr': row['ctr'],
                    'position': row['position']
                })
            return pd.DataFrame(data)
        else:
            st.warning("No data found for the specified period")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error fetching GSC data: {str(e)}")
        return None

# Function to analyze queries with Gemini using Query Fan-Out methodology
def analyze_queries_with_gemini(df, api_key, max_queries, include_branded, sort_metric):
    """Analyze queries using Gemini API with Query Fan-Out approach"""
    
    if not api_key:
        st.error("Please provide Gemini API key")
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    # Filter and prepare queries
    df_filtered = df.copy()
    if not include_branded:
        # Simple branded query filter (you can make this more sophisticated)
        df_filtered = df_filtered[~df_filtered['query'].str.contains('your brand|company name', case=False, na=False)]
    
    # Get top queries by selected metric
    if sort_metric == 'position':
        # For position, lower is better
        top_queries = df_filtered.nsmallest(max_queries, 'position')
    else:
        # For other metrics, higher is better
        top_queries = df_filtered.nlargest(max_queries, sort_metric)
    
    # Query Fan-Out Analysis Prompt
    prompt = f"""
    You are an SEO expert specializing in Query Fan-Out analysis for Google's AI Mode search. 
    
    Analyze these search queries from Google Search Console and provide a comprehensive Query Fan-Out analysis:
    
    QUERIES DATA:
    {top_queries[['query', 'impressions', 'clicks', 'position']].to_string()}
    
    Please provide a detailed analysis following the Query Fan-Out methodology:
    
    1. **PRIMARY ENTITY IDENTIFICATION**
       - Identify the main ontological entities for each query
       - Group queries by semantic intent and topic clusters
    
    2. **QUERY FAN-OUT MAPPING**
       For each primary query, identify:
       - Sub-queries that Google AI might generate
       - Related questions users might ask
       - Contextual expansions of the query
    
    3. **CONTENT COVERAGE ANALYSIS**
       - Which fan-out queries are likely covered by existing content?
       - Which represent content gaps?
       - Coverage score for each query cluster
    
    4. **AI MODE OPTIMIZATION RECOMMENDATIONS**
       - Specific content pieces to create (with titles and structure)
       - How to structure content for passage-level extraction
       - Semantic relationships to establish between pages
       - Schema markup recommendations
    
    5. **FOLLOW-UP QUERY PREDICTIONS**
       - Next likely queries in the user journey
       - How to optimize for query chains
    
    6. **PRIORITY RANKING**
       - Rank content opportunities by potential impact
       - Consider search volume, competition, and AI Mode compatibility
    
    Format your response with clear headers and actionable insights. Focus on practical recommendations that can improve visibility in AI-powered search results.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error with Gemini API: {str(e)}")
        return None

# Main app logic
if authenticate_google():
    st.success("✅ Authenticated with Google Search Console")
    
    # Get list of verified sites
    if st.button("🔄 Load Search Console Properties"):
        try:
            service = build('searchconsole', 'v1', credentials=st.session_state.credentials)
            sites = service.sites().list().execute()
            
            if 'siteEntry' in sites:
                st.session_state.sites = [site['siteUrl'] for site in sites['siteEntry']]
            else:
                st.warning("No verified sites found in your Search Console")
                st.session_state.sites = []
        except Exception as e:
            st.error(f"Error loading sites: {str(e)}")
    
    # Site selection
    if hasattr(st.session_state, 'sites') and st.session_state.sites:
        selected_site = st.selectbox("Select a property:", st.session_state.sites)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Fetch GSC Data", type="primary"):
                with st.spinner("Fetching data from Google Search Console..."):
                    df = get_gsc_data(selected_site, days_back)
                    
                    if df is not None and not df.empty:
                        # Filter by minimum value based on selected metric
                        if filter_column == "position":
                            # For position, filter by maximum (lower is better)
                            df_filtered = df[df[filter_column] <= min_value]
                        else:
                            # For other metrics, filter by minimum
                            df_filtered = df[df[filter_column] >= min_value]
                        
                        st.session_state.gsc_data = df_filtered
                        st.success(f"✅ Loaded {len(df_filtered)} queries")
                    else:
                        st.error("No data retrieved")
        
        # Display data and analysis
        if hasattr(st.session_state, 'gsc_data'):
            st.subheader("📈 Query Performance Overview")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Queries", len(st.session_state.gsc_data))
            with col2:
                st.metric("Total Clicks", f"{st.session_state.gsc_data['clicks'].sum():,}")
            with col3:
                st.metric("Total Impressions", f"{st.session_state.gsc_data['impressions'].sum():,}")
            with col4:
                avg_position = st.session_state.gsc_data['position'].mean()
                st.metric("Avg Position", f"{avg_position:.1f}")
            
            # Query distribution visualization
            st.subheader("🔍 Query Analysis")
            
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["Top Queries", "Query Distribution", "Position Analysis"])
            
            with tab1:
                # Sort by selected metric
                if sort_metric == 'position':
                    sorted_data = st.session_state.gsc_data.nsmallest(20, sort_metric)
                else:
                    sorted_data = st.session_state.gsc_data.nlargest(20, sort_metric)
                
                st.markdown(f"**Top 20 Queries by {sort_metric.upper()}**")
                st.dataframe(
                    sorted_data,
                    use_container_width=True
                )
            
            with tab2:
                # Simple distribution chart
                if sort_metric == 'position':
                    chart_data = st.session_state.gsc_data.nsmallest(15, sort_metric).set_index('query')[sort_metric]
                else:
                    chart_data = st.session_state.gsc_data.nlargest(15, sort_metric).set_index('query')[sort_metric]
                
                st.markdown(f"**Query Distribution by {sort_metric.upper()}**")
                st.bar_chart(chart_data)
            
            with tab3:
                # Position buckets
                position_buckets = pd.cut(
                    st.session_state.gsc_data['position'],
                    bins=[0, 3, 10, 20, 50, 100],
                    labels=['Top 3', '4-10', '11-20', '21-50', '50+']
                )
                st.bar_chart(position_buckets.value_counts())
            
            # AI Analysis with Query Fan-Out
            if gemini_api_key:
                if st.button("🤖 Run Query Fan-Out Analysis", type="primary"):
                    with st.spinner("Analyzing queries with Gemini AI using Query Fan-Out methodology..."):
                        analysis = analyze_queries_with_gemini(
                            st.session_state.gsc_data,
                            gemini_api_key,
                            max_queries,
                            include_branded,
                            sort_metric
                        )
                        
                        if analysis:
                            st.subheader("🎯 Query Fan-Out Analysis Results")
                            st.markdown(analysis)
                            
                            # Option to download analysis
                            st.download_button(
                                label="📥 Download Analysis Report",
                                data=analysis,
                                file_name=f"query_fanout_analysis_{selected_site.replace('https://', '').replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown"
                            )
            else:
                st.warning("⚠️ Please provide Gemini API key to enable Query Fan-Out analysis")
    
    # Logout button
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.credentials = None
        st.rerun()

else:
    st.info("👆 Please authenticate with Google to continue")
    
# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ by SEOptimize LLC | Powered by Query Fan-Out methodology</p>
        <p><a href='https://github.com/SEOptimize-LLC/SEO-Query-Fan-Out-Tool'>GitHub Repository</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
