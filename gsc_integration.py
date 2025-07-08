"""
Google Search Console Integration for Query Fan-Out Analyzer
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time
import streamlit as st

class GSCClient:
    """Google Search Console API client with rate limiting and error handling"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        self.service = None
        self.credentials_path = credentials_path
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.max_retries = 3
        
    def authenticate(self, credentials_json: Optional[Dict] = None):
        """Authenticate with Google Search Console API"""
        try:
            if credentials_json:
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_json,
                    scopes=['https://www.googleapis.com/auth/webmasters.readonly']
                )
            elif self.credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=['https://www.googleapis.com/auth/webmasters.readonly']
                )
            else:
                raise ValueError("No credentials provided")
            
            self.service = build('searchconsole', 'v1', credentials=credentials)
            return True
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")
            return False
    
    def get_sites(self) -> List[str]:
        """Get list of verified sites"""
        try:
            sites = self.service.sites().list().execute()
            return [site['siteUrl'] for site in sites.get('siteEntry', [])]
        except HttpError as e:
            st.error(f"Error fetching sites: {str(e)}")
            return []
    
    def get_search_analytics(
        self,
        site_url: str,
        start_date: str,
        end_date: str,
        dimensions: List[str] = ['query', 'page'],
        row_limit: int = 25000,
        start_row: int = 0
    ) -> pd.DataFrame:
        """Get search analytics data from GSC"""
        
        request_body = {
            'startDate': start_date,
            'endDate': end_date,
            'dimensions': dimensions,
            'rowLimit': row_limit,
            'startRow': start_row,
            'dataState': 'final'
        }
        
        all_rows = []
        
        while True:
            try:
                response = self._execute_with_retry(
                    self.service.searchanalytics().query(
                        siteUrl=site_url,
                        body=request_body
                    )
                )
                
                if 'rows' not in response:
                    break
                
                rows = response['rows']
                all_rows.extend(rows)
                
                # Check if we have more data
                if len(rows) < row_limit:
                    break
                
                # Update start row for next batch
                request_body['startRow'] += row_limit
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except HttpError as e:
                if e.resp.status == 429:  # Rate limit exceeded
                    st.warning("Rate limit reached, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                else:
                    st.error(f"Error fetching data: {str(e)}")
                    break
        
        # Convert to DataFrame
        if all_rows:
            df = self._process_response_to_dataframe(all_rows, dimensions)
            return df
        else:
            return pd.DataFrame()
    
    def _execute_with_retry(self, request):
        """Execute request with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                return request.execute()
            except HttpError as e:
                if e.resp.status == 429 and attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # Exponential backoff
                    time.sleep(wait_time)
                else:
                    raise
    
    def _process_response_to_dataframe(self, rows: List[Dict], dimensions: List[str]) -> pd.DataFrame:
        """Process GSC response into DataFrame"""
        data = []
        
        for row in rows:
            item = {}
            
            # Add dimensions
            for i, dimension in enumerate(dimensions):
                item[dimension] = row['keys'][i]
            
            # Add metrics
            item['clicks'] = row.get('clicks', 0)
            item['impressions'] = row.get('impressions', 0)
            item['ctr'] = row.get('ctr', 0)
            item['position'] = row.get('position', 0)
            
            data.append(item)
        
        return pd.DataFrame(data)
    
    def get_query_data_for_urls(
        self,
        site_url: str,
        urls: List[str],
        start_date: str,
        end_date: str,
        batch_size: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """Get query data for specific URLs"""
        results = {}
        
        # Process URLs in batches
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i + batch_size]
            
            for url in batch_urls:
                request_body = {
                    'startDate': start_date,
                    'endDate': end_date,
                    'dimensions': ['query'],
                    'dimensionFilterGroups': [{
                        'filters': [{
                            'dimension': 'page',
                            'operator': 'equals',
                            'expression': url
                        }]
                    }],
                    'rowLimit': 1000
                }
                
                try:
                    response = self._execute_with_retry(
                        self.service.searchanalytics().query(
                            siteUrl=site_url,
                            body=request_body
                        )
                    )
                    
                    if 'rows' in response:
                        df = self._process_response_to_dataframe(
                            response['rows'],
                            ['query']
                        )
                        results[url] = df
                    else:
                        results[url] = pd.DataFrame()
                    
                    # Rate limiting
                    time.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    st.warning(f"Error fetching data for {url}: {str(e)}")
                    results[url] = pd.DataFrame()
            
            # Show progress
            progress = min((i + batch_size) / len(urls), 1.0)
            st.progress(progress)
        
        return results
    
    def analyze_query_patterns(self, query_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze query patterns for fan-out insights"""
        if query_data.empty:
            return {}
        
        analysis = {
            'total_queries': len(query_data),
            'total_clicks': query_data['clicks'].sum(),
            'total_impressions': query_data['impressions'].sum(),
            'avg_position': query_data['position'].mean(),
            'avg_ctr': query_data['ctr'].mean()
        }
        
        # Query type analysis
        query_types = {
            'what': 0,
            'how': 0,
            'why': 0,
            'when': 0,
            'where': 0,
            'best': 0,
            'guide': 0,
            'tutorial': 0
        }
        
        for query in query_data['query']:
            query_lower = query.lower()
            for query_type in query_types:
                if query_type in query_lower:
                    query_types[query_type] += 1
        
        analysis['query_types'] = query_types
        
        # Top performing queries
        top_queries = query_data.nlargest(10, 'clicks')[['query', 'clicks', 'impressions', 'ctr', 'position']]
        analysis['top_queries'] = top_queries
        
        # Low hanging fruit (high impressions, low position)
        low_hanging = query_data[
            (query_data['impressions'] > query_data['impressions'].quantile(0.75)) &
            (query_data['position'] > 10)
        ].nlargest(10, 'impressions')
        analysis['optimization_opportunities'] = low_hanging
        
        return analysis

def integrate_gsc_data(
    gsc_client: GSCClient,
    site_url: str,
    analysis_results: List[Dict[str, Any]],
    date_range: int = 28
) -> List[Dict[str, Any]]:
    """Integrate GSC data with analysis results"""
    
    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=date_range)
    
    # Get URLs from analysis results
    urls = [result['url'] for result in analysis_results if result['status'] == 'success']
    
    if not urls:
        return analysis_results
    
    # Fetch query data for URLs
    st.info(f"Fetching GSC data for {len(urls)} URLs...")
    query_data = gsc_client.get_query_data_for_urls(
        site_url,
        urls,
        start_date.isoformat(),
        end_date.isoformat()
    )
    
    # Merge with analysis results
    for result in analysis_results:
        if result['url'] in query_data:
            url_queries = query_data[result['url']]
            
            if not url_queries.empty:
                # Add GSC metrics
                result['gsc_total_queries'] = len(url_queries)
                result['gsc_total_clicks'] = url_queries['clicks'].sum()
                result['gsc_total_impressions'] = url_queries['impressions'].sum()
                result['gsc_avg_position'] = url_queries['position'].mean()
                result['gsc_avg_ctr'] = url_queries['ctr'].mean()
                
                # Add top queries
                top_queries = url_queries.nlargest(5, 'clicks')['query'].tolist()
                result['gsc_top_queries'] = top_queries
                
                # Query diversity score
                result['query_diversity_score'] = min(len(url_queries) / 50, 1.0)
            else:
                result['gsc_data_available'] = False
        else:
            result['gsc_data_available'] = False
    
    return analysis_results

def create_gsc_dashboard(gsc_data: pd.DataFrame):
    """Create GSC data dashboard in Streamlit"""
    st.subheader("Google Search Console Insights")
    
    if gsc_data.empty:
        st.warning("No GSC data available")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", f"{len(gsc_data):,}")
    with col2:
        st.metric("Total Clicks", f"{gsc_data['clicks'].sum():,}")
    with col3:
        st.metric("Avg CTR", f"{gsc_data['ctr'].mean():.2%}")
    with col4:
        st.metric("Avg Position", f"{gsc_data['position'].mean():.1f}")
    
    # Query performance chart
    st.markdown("---")
    
    import plotly.express as px
    
    # Top queries by clicks
    top_queries = gsc_data.nlargest(20, 'clicks')
    
    fig = px.bar(
        top_queries,
        x='clicks',
        y='query',
        orientation='h',
        title='Top 20 Queries by Clicks',
        labels={'clicks': 'Clicks', 'query': 'Search Query'}
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # CTR vs Position scatter plot
    fig_scatter = px.scatter(
        gsc_data[gsc_data['impressions'] > 100],  # Filter low impression queries
        x='position',
        y='ctr',
        size='impressions',
        hover_data=['query', 'clicks'],
        title='CTR vs Position Analysis',
        labels={'position': 'Average Position', 'ctr': 'Click-Through Rate'}
    )
    fig_scatter.update_yaxis(tickformat='.1%')
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Opportunity identification
    st.markdown("---")
    st.subheader("Optimization Opportunities")
    
    # High impressions, low CTR
    low_ctr = gsc_data[
        (gsc_data['impressions'] > gsc_data['impressions'].quantile(0.7)) &
        (gsc_data['ctr'] < gsc_data['ctr'].quantile(0.3))
    ].nlargest(10, 'impressions')
    
    if not low_ctr.empty:
        st.markdown("### 🎯 Low CTR Opportunities")
        st.markdown("High impression queries with low CTR - optimize meta descriptions and titles")
        st.dataframe(
            low_ctr[['query', 'impressions', 'clicks', 'ctr', 'position']],
            use_container_width=True
        )
    
    # Position 11-20 opportunities
    position_opp = gsc_data[
        (gsc_data['position'] >= 11) & 
        (gsc_data['position'] <= 20) &
        (gsc_data['impressions'] > 100)
    ].nlargest(10, 'impressions')
    
    if not position_opp.empty:
        st.markdown("### 📈 Page 2 Opportunities")
        st.markdown("Queries ranking on page 2 - small improvements could boost to page 1")
        st.dataframe(
            position_opp[['query', 'impressions', 'clicks', 'ctr', 'position']],
            use_container_width=True
        )
