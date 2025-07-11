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
        model_name = settings.get('gemini_model', 'gemini-1.5-flash')
        model = genai.GenerativeModel(model_name)
        
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
    def analyze_query_fanout_manual(queries_df, api_key, analysis_settings):
        """
        Perform Query Fan-Out analysis for manual queries (new content planning)
        
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
        model_name = analysis_settings.get('gemini_model', 'gemini-1.5-flash')
        model = genai.GenerativeModel(model_name)
        
        # Build the analysis prompt for new content
        queries_list = queries_df['query'].tolist()
        
        depth_instructions = {
            "Basic": "Provide a concise analysis focusing on the main fan-out queries and content structure.",
            "Standard": "Provide a detailed analysis including sub-queries, content recommendations, and implementation tips.",
            "Comprehensive": "Provide an exhaustive analysis including all query expansions, detailed content strategies, technical SEO requirements, and step-by-step implementation guide."
        }
        
        prompt = f"""
        You are an expert in Google's Query Fan-Out methodology and AI-powered search optimization.
        
        OPTIMIZATION TARGET: {'Google AI Overviews (Simple, direct answers)' if analysis_settings.get('ai_search_type') == 'ai_overviews' else 'Google AI Mode (Complex query fan-out)'}
        
        Analyze these queries for NEW CONTENT CREATION:
        
        TARGET QUERIES FOR NEW CONTENT:
        {chr(10).join(f"- {q}" for q in queries_list)}
        
        ANALYSIS DEPTH: {depth_instructions[analysis_settings.get('depth', 'Standard')]}
        """
        
        if analysis_settings.get('ai_search_type') == 'ai_overviews':
            # AI Overviews optimization for new content
            prompt += """
        
        Please provide a content creation strategy for AI OVERVIEWS optimization:
        
        1. **DIRECT ANSWER CONTENT STRATEGY**
           For each target query:
           - The exact 40-60 word answer to include at the top
           - Supporting details and context
           - Definition boxes and quick facts
           - Scannable format recommendations
        
        2. **SNIPPET-FIRST CONTENT ARCHITECTURE**
           - How to structure content for featured snippets
           - Paragraph vs. list vs. table format for each query
           - HTML markup for optimal extraction
           - Jump links and navigation
        
        3. **QUICK ANSWER OPTIMIZATION**
           - First 100 words optimization for each piece
           - Clear headers and subheaders
           - Bulleted lists and numbered steps
           - Summary boxes and key takeaways
        
        4. **SUPPORTING CONTENT ELEMENTS**
           - FAQs to include
           - Quick reference sections
           - Glossary terms
           - Comparison tables
        """
            
            if analysis_settings.get('include_snippet_optimization'):
                prompt += """
        
        5. **FEATURED SNIPPET TEMPLATES**
           Provide exact snippet-optimized content for top queries:
           - Definition snippets (What is...)
           - List snippets (Types of..., Steps to...)
           - Table snippets (Comparison, features)
           - Paragraph snippets (How to..., Why...)
        """
        
        else:
            # AI Mode optimization for new content (Complex)
            prompt += """
        
        Please provide a comprehensive content strategy for AI MODE optimization:
        
        1. **QUERY FAN-OUT MAPPING FOR NEW CONTENT**
           For each target query:
           - Primary intent and all sub-intents
           - Complete list of fan-out queries AI would generate
           - Multi-step research journeys
           - Entity relationships to establish
        
        2. **COMPREHENSIVE CONTENT ARCHITECTURE**
           - Topic cluster structure
           - Pillar page vs. supporting pages
           - Content depth requirements (2000-5000+ words)
           - Semantic coverage checklist
           - Internal linking blueprint
        
        3. **AI MODE OPTIMIZATION BLUEPRINT**
           - Passage-level optimization strategy
           - Entity markup throughout content
           - Semantic HTML structure
           - Progressive disclosure techniques
           - Multi-format content (text, lists, tables, media)
        """
            
            if analysis_settings.get('include_followup'):
                prompt += """
        
        4. **FOLLOW-UP QUERY CONTENT MAPPING**
           - Anticipated user journeys
           - Next-step content recommendations
           - Decision trees and flowcharts
           - Related topics to cover
           - Cross-linking opportunities
        """
            
            if analysis_settings.get('include_entity_mapping'):
                prompt += """
        
        5. **ENTITY-BASED CONTENT STRATEGY**
           - Core entities to define and explain
           - Entity relationship diagrams
           - Knowledge base structure
           - Semantic markup plan
           - Glossary and definition sections
        """
        
        # Common sections for both types
        prompt += """
        
        6. **CONTENT CREATION ROADMAP**
           Prioritized implementation plan:
           - Which content to create first
           - Dependencies and prerequisites
           - Estimated effort and impact
           - Publishing schedule
        
        7. **SUCCESS METRICS**
           - Target rankings for each query
           - Expected CTR improvements
           - Engagement metrics to track
           - AI visibility indicators
        """
        
        if analysis_settings.get('include_schema', True):
            prompt += """
        
        5. **SCHEMA MARKUP STRATEGY**
           - Essential schema types for each content piece
           - Properties to maximize AI understanding
           - FAQ, HowTo, and other relevant schemas
           - Entity markup recommendations
        """
        
        if analysis_settings.get('include_competitors', False):
            prompt += """
        
        6. **COMPETITIVE CONTENT ANALYSIS**
           - What competitors likely rank for these queries
           - Content gaps to exploit
           - Unique angles and differentiation strategies
           - How to create 10x better content
        """
        
        prompt += """
        
        7. **CONTENT CREATION ROADMAP**
           Provide a prioritized implementation plan:
           - Which content to create first (quick wins)
           - Content dependencies and optimal publishing order
           - Estimated effort and impact for each piece
           - Success metrics to track
        
        8. **CONTENT BRIEF TEMPLATES**
           For the top 3 priority pieces, provide:
           - Target keyword cluster
           - Content outline with word count targets
           - Key points to cover
           - Unique value proposition
        
        Format your response with clear, actionable recommendations for content creation.
        Focus on practical implementation for maximum AI search visibility.
        """
        
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error during Gemini analysis: {str(e)}")
            return None
        """Build the prompt for Gemini analysis"""
        
        # Format query data
        query_data = queries_df[['query', 'clicks', 'impressions', 'ctr', 'position']].to_string(index=False)
        
        prompt = f"""
        You are an expert in Google's Query Fan-Out methodology and AI-powered search optimization.
        
        OPTIMIZATION TARGET: {'Google AI Overviews (Simple, direct answers)' if settings.get('ai_search_type') == 'ai_overviews' else 'Google AI Mode (Complex query fan-out)'}
        
        Analyze these Google Search Console queries and provide optimization recommendations:
        
        QUERY PERFORMANCE DATA:
        {query_data}
        
        ANALYSIS PARAMETERS:
        - Focus on queries sorted by: {settings.get('sort_metric', 'clicks')}
        - Analysis depth: {settings.get('depth', 'comprehensive')}
        - AI Search Type: {settings.get('ai_search_type', 'ai_mode')}
        """
        
        if settings.get('ai_search_type') == 'ai_overviews':
            # AI Overviews optimization (Simple)
            prompt += """
        
        Please provide analysis for AI OVERVIEWS optimization:
        
        1. **DIRECT ANSWER OPTIMIZATION**
           - Which queries need clear, concise answers in the first paragraph
           - Ideal answer length (40-60 words) for each query
           - Definition-style formatting recommendations
           - List and table opportunities
        
        2. **SNIPPET OPTIMIZATION STRATEGY**
           - Featured snippet opportunities for each query
           - Paragraph vs. list vs. table snippet recommendations
           - Optimal formatting for quick extraction
           - Answer box targeting techniques
        
        3. **CONTENT STRUCTURE FOR AI OVERVIEWS**
           - Lead paragraph optimization for each topic
           - Clear question-answer formatting
           - Scannable content structure
           - Bold text and emphasis strategies
        """
            
            if settings.get('include_snippet_optimization'):
                prompt += """
        
        4. **FEATURED SNIPPET TARGETING**
           - Specific snippet formats for each query type
           - Character/word count recommendations
           - HTML markup for better extraction
           - Common snippet trigger patterns
        """
            
            if settings.get('include_paa_optimization'):
                prompt += """
        
        5. **PEOPLE ALSO ASK OPTIMIZATION**
           - Related questions to include for each topic
           - Q&A schema implementation
           - Accordion/expandable content recommendations
           - PAA box targeting strategies
        """
        
        else:
            # AI Mode optimization (Complex)
            prompt += """
        
        Please provide analysis for AI MODE optimization (Complex Query Fan-Out):
        
        1. **PRIMARY ENTITY & INTENT MAPPING**
           - Identify the main ontological entities for each query
           - Classify query intent (informational, transactional, navigational, commercial)
           - Group queries into semantic clusters
           - Knowledge graph connections
        
        2. **QUERY FAN-OUT PREDICTIONS**
           For each primary query, identify:
           - ALL sub-queries that Google AI Mode would generate
           - Multi-step research paths users might follow
           - Contextual expansions and refinements
           - Related entity queries
        
        3. **COMPREHENSIVE CONTENT COVERAGE**
           - Which fan-out queries need dedicated sections
           - Content depth requirements for AI Mode
           - Topic cluster architecture
           - Semantic completeness scoring
        """
            
            if settings.get('include_followup'):
                prompt += """
        
        4. **FOLLOW-UP QUERY MAPPING**
           - Predict next queries in user journey
           - Multi-hop question sequences
           - Decision tree content structure
           - Progressive disclosure strategies
        """
            
            if settings.get('include_entity_mapping'):
                prompt += """
        
        5. **ENTITY RELATIONSHIP MAPPING**
           - Core entities and their relationships
           - Knowledge graph optimization
           - Entity markup and disambiguation
           - Semantic triple recommendations
        """
        
        # Common sections for both types
        prompt += """
        
        6. **CONTENT OPTIMIZATION PRIORITIES**
           Based on current performance:
           - Quick wins: Minimal changes for maximum impact
           - Medium-term: Content additions and restructuring  
           - Long-term: New content creation needs
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
