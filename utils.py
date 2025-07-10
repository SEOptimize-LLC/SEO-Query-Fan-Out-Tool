"""
Utility functions for Query Fan-Out Analysis
"""

import streamlit as st
import google.generativeai as genai
import pandas as pd
from datetime import datetime
import re
from config import Config


class QueryAnalyzer:
    """Handle query analysis and fan-out predictions"""
    
    @staticmethod
    def filter_queries(df, min_value, filter_column, include_branded=False, brand_terms=None):
        """Filter queries based on criteria"""
        
        df_filtered = df.copy()
        
        # Apply metric filter
        if filter_column == "position":
            # For position, lower is better
            df_filtered = df_filtered[df_filtered[filter_column] <= min_value]
        else:
            # For other metrics, higher is better
            df_filtered = df_filtered[df_filtered[filter_column] >= min_value]
        
        # Filter branded queries if requested
        if not include_branded and brand_terms:
            brand_pattern = '|'.join([re.escape(term) for term in brand_terms])
            df_filtered = df_filtered[~df_filtered['query'].str.contains(brand_pattern, case=False, na=False)]
        
        return df_filtered
    
    @staticmethod
    def analyze_query_fanout(queries_df, api_key, analysis_settings):
        """
        Perform Query Fan-Out analysis using Gemini
        
        Args:
            queries_df: DataFrame with query data
            api_key: Gemini API key
            analysis_settings: dict with analysis parameters
        """
        
        if not api_key:
            st.error("Please provide a Gemini API key")
            return None
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Prepare query data
        max_queries = analysis_settings.get('max_queries', 20)
        sort_metric = analysis_settings.get('sort_metric', 'clicks')
        
        # Sort queries appropriately
        if sort_metric == 'position':
            top_queries = queries_df.nsmallest(max_queries, sort_metric)
        else:
            top_queries = queries_df.nlargest(max_queries, sort_metric)
        
        # Build the analysis prompt
        prompt = QueryAnalyzer._build_fanout_prompt(top_queries, analysis_settings)
        
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error during Gemini analysis: {str(e)}")
            return None
    
    @staticmethod
    def _build_fanout_prompt(queries_df, settings):
        """Build the prompt for Gemini analysis"""
        
        # Format query data
        query_data = queries_df[['query', 'clicks', 'impressions', 'ctr', 'position']].to_string(index=False)
        
        prompt = f"""
        You are an expert in Google's Query Fan-Out methodology and AI-powered search optimization.
        
        Analyze these Google Search Console queries and provide a comprehensive Query Fan-Out analysis:
        
        QUERY PERFORMANCE DATA:
        {query_data}
        
        ANALYSIS PARAMETERS:
        - Focus on queries sorted by: {settings.get('sort_metric', 'clicks')}
        - Analysis depth: {settings.get('depth', 'comprehensive')}
        
        Please provide a detailed analysis following the Query Fan-Out methodology:
        
        1. **PRIMARY ENTITY & INTENT MAPPING**
           - Identify the main ontological entities for each query
           - Classify query intent (informational, transactional, navigational, commercial)
           - Group queries into semantic clusters
        
        2. **QUERY FAN-OUT PREDICTIONS**
           For each primary query, identify:
           - Sub-queries that Google AI Mode would likely generate
           - Related questions users might ask in the same session
           - Contextual expansions and refinements
           - Entity relationships and knowledge graph connections
        
        3. **CONTENT COVERAGE ASSESSMENT**
           Based on the performance metrics:
           - Which fan-out queries are likely already covered? (high CTR = good coverage)
           - Which represent content gaps? (high impressions, low CTR)
           - Coverage score for each query cluster
        
        4. **AI MODE OPTIMIZATION STRATEGY**
           Specific recommendations for:
           - Content structure for passage-level extraction
           - Entity markup and semantic HTML
           - Internal linking to establish topical authority
           - Content depth requirements for each cluster
        
        5. **QUICK WINS vs LONG-TERM OPPORTUNITIES**
           Based on current performance:
           - Quick wins: Queries with positions 4-20 (near first page)
           - Medium-term: High impression, low CTR queries
           - Long-term: New content for uncovered fan-out queries
        """
        
        # Add optional sections based on settings
        if settings.get('include_schema', True):
            prompt += """
        
        6. **SCHEMA MARKUP RECOMMENDATIONS**
           - Specific schema types for each content piece
           - Required and recommended properties
           - How schema enhances AI understanding
        """
        
        if settings.get('include_competitors', False):
            prompt += """
        
        7. **COMPETITIVE LANDSCAPE ANALYSIS**
           - Likely competitor strategies for these queries
           - Differentiation opportunities
           - Content gaps in the market
        """
        
        prompt += """
        
        8. **IMPLEMENTATION ROADMAP**
           Provide a prioritized action plan:
           - Week 1-2: Immediate optimizations
           - Month 1: Content updates and additions
           - Month 2-3: New content creation
           - Ongoing: Monitoring and iteration
        
        Format your response with clear headers, specific examples, and actionable recommendations.
        Focus on practical implementation rather than theory.
        """
        
        return prompt
    
    @staticmethod
    def export_analysis(analysis_text, queries_df, format='markdown'):
        """Export analysis in various formats"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if format == 'markdown':
            return f"""# Query Fan-Out Analysis Report
Generated: {timestamp}

## Summary Statistics
- Total Queries Analyzed: {len(queries_df)}
- Total Clicks: {queries_df['clicks'].sum():,}
- Total Impressions: {queries_df['impressions'].sum():,}
- Average CTR: {queries_df['ctr'].mean():.2%}
- Average Position: {queries_df['position'].mean():.1f}

## Top Performing Queries
{queries_df.head(10).to_markdown(index=False)}

## Query Fan-Out Analysis
{analysis_text}

---
*Report generated by Query Fan-Out Analysis Tool*
"""
        
        elif format == 'csv':
            # Return queries with analysis appended
            queries_export = queries_df.copy()
            queries_export['analysis_date'] = timestamp
            return queries_export
        
        else:
            return analysis_text


class UIHelpers:
    """Helper functions for Streamlit UI"""
    
    @staticmethod
    def display_metrics(df):
        """Display summary metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", f"{len(df):,}")
        
        with col2:
            st.metric("Total Clicks", f"{df['clicks'].sum():,}")
        
        with col3:
            st.metric("Avg CTR", f"{df['ctr'].mean():.2%}")
        
        with col4:
            st.metric("Avg Position", f"{df['position'].mean():.1f}")
    
    @staticmethod
    def create_position_distribution(df):
        """Create position distribution chart data"""
        bins = [0, 3, 10, 20, 50, 100]
        labels = ['Top 3', 'Page 1 (4-10)', 'Page 2', 'Page 3-5', 'Beyond']
        
        df['position_range'] = pd.cut(df['position'], bins=bins, labels=labels, include_lowest=True)
        return df['position_range'].value_counts().sort_index()
    
    @staticmethod
    def highlight_opportunities(df):
        """Identify and highlight optimization opportunities"""
        opportunities = []
        
        # High impressions, low CTR
        high_imp_low_ctr = df[(df['impressions'] > df['impressions'].quantile(0.7)) & 
                              (df['ctr'] < df['ctr'].quantile(0.3))]
        if not high_imp_low_ctr.empty:
            opportunities.append({
                'type': 'High Impressions, Low CTR',
                'queries': high_imp_low_ctr.head(5)['query'].tolist(),
                'action': 'Improve meta descriptions and title tags'
            })
        
        # Good CTR, poor position
        good_ctr_poor_pos = df[(df['ctr'] > df['ctr'].quantile(0.7)) & 
                               (df['position'] > 10)]
        if not good_ctr_poor_pos.empty:
            opportunities.append({
                'type': 'Good CTR, Poor Position',
                'queries': good_ctr_poor_pos.head(5)['query'].tolist(),
                'action': 'Boost content quality and internal linking'
            })
        
        # Near first page (positions 11-20)
        near_first_page = df[(df['position'] > 10) & (df['position'] <= 20)]
        if not near_first_page.empty:
            opportunities.append({
                'type': 'Near First Page',
                'queries': near_first_page.head(5)['query'].tolist(),
                'action': 'Quick wins - minor optimizations can push to page 1'
            })
        
        return opportunities
