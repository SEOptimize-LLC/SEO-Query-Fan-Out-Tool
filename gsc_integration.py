"""
Google Search Console Integration Module
Handles OAuth authentication and data retrieval from GSC
"""

import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import pandas as pd
from config import Config


class GSCAuth:
    """Handle Google Search Console OAuth authentication"""
    
    @staticmethod
    def authenticate():
        """Handle the OAuth flow for Google Search Console"""
        
        client_id, client_secret = Config.get_google_credentials()
        
        if not client_id or not client_secret:
            st.error("❌ Google OAuth credentials not found. Please configure them in settings.")
            return False
        
        redirect_uri = Config.get_redirect_uri()
        
        # Debug info
        with st.expander("🔧 OAuth Debug Info", expanded=True):
            st.write(f"**Redirect URI being used:** `{redirect_uri}`")
            st.write(f"**Client ID:** `{client_id[:30]}...` (truncated for security)")
            st.info("**Important:** This exact redirect URI must be added to your Google Cloud Console OAuth settings!")
            st.write("**To fix authentication issues:**")
            st.write("1. Copy the redirect URI above")
            st.write("2. Go to Google Cloud Console → APIs & Services → Credentials")
            st.write("3. Edit your OAuth 2.0 Client ID")
            st.write("4. Add this exact URI to 'Authorized redirect URIs'")
            st.write("5. Save and try again")
        
        # Create flow
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [redirect_uri]
                }
            },
            scopes=Config.SCOPES
        )
        
        flow.redirect_uri = redirect_uri
        
        # Check for authorization code in URL
        auth_code = st.query_params.get("code", None)
        
        if auth_code and not st.session_state.get('authenticated', False):
            try:
                # Exchange code for token
                flow.fetch_token(code=auth_code)
                st.session_state.credentials = flow.credentials
                st.session_state.authenticated = True
                
                # Clear URL parameters
                st.query_params.clear()
                
                st.success("✅ Successfully authenticated with Google!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Authentication failed: {str(e)}")
                if "redirect_uri_mismatch" in str(e):
                    st.error("**Redirect URI mismatch!** Make sure the redirect URI in Google Cloud Console matches exactly.")
                return False
        
        elif not st.session_state.get('authenticated', False):
            # Generate auth URL with account selection
            auth_url, _ = flow.authorization_url(
                prompt="select_account",
                access_type="offline",
                include_granted_scopes="true"
            )
            
            st.markdown("### 🔐 Google Authentication Required")
            st.markdown("Click the button below to authenticate with your Google Search Console account:")
            
            # Create a more prominent button that opens in a new tab
            st.markdown(
                f"""
                <a href="{auth_url}" target="_blank" style="text-decoration: none;">
                    <button style="
                        background-color: #4285F4;
                        color: white;
                        padding: 12px 24px;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 16px;
                        font-weight: 500;
                        display: inline-flex;
                        align-items: center;
                        gap: 8px;
                        margin: 10px 0;
                    ">
                        <svg width="18" height="18" viewBox="0 0 48 48" style="display: inline;">
                            <path fill="#4285F4" d="M44.5 20H24v8.5h11.8C34.7 33.9 30.1 37 24 37c-7.2 0-13-5.8-13-13s5.8-13 13-13c3.1 0 5.9 1.1 8.1 2.9l6.4-6.4C34.6 4.1 29.6 2 24 2 11.8 2 2 11.8 2 24s9.8 22 22 22c11 0 21-8 21-22 0-1.3-.2-2.7-.5-4z"/>
                            <path fill="#34A853" d="M6.3 14.7l7 5.1C15 16 19.1 13 24 13c3.1 0 5.9 1.1 8.1 2.9l6.4-6.4C34.6 4.1 29.6 2 24 2 15.4 2 7.9 6.7 4.6 13.5l1.7 1.2z"/>
                            <path fill="#FBBC05" d="M24 46c5.4 0 10.2-1.8 14-5.1l-6.8-5.7C29.1 36.4 26.7 37 24 37c-6 0-10.5-3-12.2-8.3l-7.3 5.6C7.9 41.2 15.4 46 24 46z"/>
                            <path fill="#EA4335" d="M44.5 20H24v8.5h11.8C34.9 31.7 32.5 34.2 29.2 35.2l6.8 5.7c3.9-3.6 6.5-8.8 6.5-14.9 0-1.3-.2-2.7-.5-4z"/>
                        </svg>
                        Sign in with Google
                    </button>
                </a>
                """,
                unsafe_allow_html=True
            )
            
            st.info("ℹ️ Opens in a new tab. After authentication, return to this tab and the app will update automatically.")
            
            # Add manual refresh button in case auto-refresh fails
            if st.button("🔄 Check Authentication Status"):
                st.rerun()
            return False
        
        return True
    
    @staticmethod
    def logout():
        """Clear authentication"""
        st.session_state.authenticated = False
        st.session_state.credentials = None
        if 'sites' in st.session_state:
            del st.session_state.sites
        if 'gsc_data' in st.session_state:
            del st.session_state.gsc_data


class GSCData:
    """Handle Google Search Console data operations"""
    
    @staticmethod
    def get_sites():
        """Get list of verified sites from GSC"""
        if not st.session_state.get('credentials'):
            return []
        
        try:
            service = build('searchconsole', 'v1', credentials=st.session_state.credentials)
            sites_list = service.sites().list().execute()
            
            if 'siteEntry' in sites_list:
                return [site['siteUrl'] for site in sites_list['siteEntry']]
            return []
            
        except Exception as e:
            st.error(f"Error fetching sites: {str(e)}")
            return []
    
    @staticmethod
    def fetch_query_data(site_url, days_back=30, dimensions=['query']):
        """Fetch search analytics data from GSC"""
        
        if not st.session_state.get('credentials'):
            st.error("Not authenticated")
            return None
        
        try:
            service = build('searchconsole', 'v1', credentials=st.session_state.credentials)
            
            # Calculate date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            # Build request
            request_body = {
                'startDate': start_date.strftime('%Y-%m-%d'),
                'endDate': end_date.strftime('%Y-%m-%d'),
                'dimensions': dimensions,
                'rowLimit': 1000,
                'dataState': 'all'
            }
            
            # Execute request
            response = service.searchanalytics().query(
                siteUrl=site_url,
                body=request_body
            ).execute()
            
            # Convert to DataFrame
            if 'rows' in response:
                data = []
                for row in response['rows']:
                    item = {
                        'query': row['keys'][0],
                        'clicks': row['clicks'],
                        'impressions': row['impressions'],
                        'ctr': row['ctr'],
                        'position': row['position']
                    }
                    data.append(item)
                
                df = pd.DataFrame(data)
                
                # Calculate additional metrics
                df['click_potential'] = df['impressions'] - df['clicks']
                df['visibility_score'] = (df['impressions'] * (1 / df['position'])).round(2)
                
                return df
            else:
                st.warning("No data found for the specified period")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error fetching GSC data: {str(e)}")
            if "HttpError 403" in str(e):
                st.error("Permission denied. Make sure you have access to this Search Console property.")
            return None
    
    @staticmethod
    def get_top_pages(site_url, days_back=30):
        """Fetch top performing pages from GSC"""
        
        if not st.session_state.get('credentials'):
            return None
        
        try:
            service = build('searchconsole', 'v1', credentials=st.session_state.credentials)
            
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            request_body = {
                'startDate': start_date.strftime('%Y-%m-%d'),
                'endDate': end_date.strftime('%Y-%m-%d'),
                'dimensions': ['page'],
                'rowLimit': 100,
                'dataState': 'all'
            }
            
            response = service.searchanalytics().query(
                siteUrl=site_url,
                body=request_body
            ).execute()
            
            if 'rows' in response:
                data = []
                for row in response['rows']:
                    data.append({
                        'page': row['keys'][0],
                        'clicks': row['clicks'],
                        'impressions': row['impressions'],
                        'ctr': row['ctr'],
                        'position': row['position']
                    })
                return pd.DataFrame(data)
            
            return pd.DataFrame()
            
        except Exception as e:
            st.error(f"Error fetching page data: {str(e)}")
            return None
