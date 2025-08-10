"""
Utility functions for Query Fan-Out Analysis
"""

import streamlit as st
import google.generativeai as genai
import pandas as pd
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
from config import Config


class QueryAnalyzer:
    """Handle query analysis and fan-out predictions"""
    
    @staticmethod
    def analyze_query_fanout_new_content(queries_df, api_key, analysis_settings):
        """
        Perform Query Fan-Out analysis for new content planning
        """
        if not api_key:
            st.error("Please provide a Gemini API key")
            return None
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        model_name = analysis_settings.get('gemini_model', 'gemini-1.5-flash')
        model = genai.GenerativeModel(model_name)
        
        # Build the analysis prompt
        queries_list = queries_df['query'].tolist()
        prompt = QueryAnalyzer._build_new_content_prompt(queries_list, analysis_settings)
        
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error during Gemini analysis: {str(e)}")
            return None
    
    @staticmethod
    def _build_new_content_prompt(queries_list, settings):
        """Build prompt for new content analysis using Query Fan-Out methodology"""
        
        # Get variant types descriptions
        variant_descriptions = []
        for vtype in settings.get('variant_types', ['equivalent', 'follow_up']):
            if vtype == 'equivalent':
                variant_descriptions.append("- Equivalent: Alternative ways to ask the same question")
            elif vtype == 'follow_up':
                variant_descriptions.append("- Follow-up: Logical next questions")
            elif vtype == 'generalization':
                variant_descriptions.append("- Generalization: Broader versions of queries")
            elif vtype == 'canonicalization':
                variant_descriptions.append("- Canonicalization: Standardized search terms")
            elif vtype == 'entailment':
                variant_descriptions.append("- Entailment: Logically implied queries")
            elif vtype == 'specification':
                variant_descriptions.append("- Specification: More detailed versions")
            elif vtype == 'clarification':
                variant_descriptions.append("- Clarification: Intent clarification queries")
        
        prompt = f"""
        You are an expert in Google's Query Fan-Out system and AI-powered search optimization.
        
        CONTEXT:
        - Target Audience: {settings.get('target_audience', 'General audience')}
        - Content Type: {settings.get('content_type', 'Blog Post')}
        - Optimization Target: {settings.get('ai_search_type', 'ai_mode').replace('_', ' ').title()}
        - Analysis Depth: {settings.get('depth', 'Standard')}
        
        QUERY VARIANT TYPES TO GENERATE:
        {chr(10).join(variant_descriptions)}
        
        TARGET QUERIES FOR NEW CONTENT:
        {chr(10).join(f"{i+1}. {q}" for i, q in enumerate(queries_list))}
        
        Please provide a comprehensive Query Fan-Out analysis following Google's methodology:
        
        ## 1. QUERY FAN-OUT GENERATION
        For each target query, generate ALL requested variant types:
        """
        
        # Add variant-specific instructions
        for vtype in settings.get('variant_types', []):
            if vtype == 'equivalent':
                prompt += """
        - **Equivalent Queries**: List 3-5 alternative phrasings users might use
        """
            elif vtype == 'follow_up':
                prompt += """
        - **Follow-up Queries**: List 3-5 logical next questions users would ask
        """
            elif vtype == 'generalization':
                prompt += """
        - **Generalization Queries**: List 2-3 broader topic queries
        """
            elif vtype == 'specification':
                prompt += """
        - **Specification Queries**: List 3-5 more specific/detailed versions
        """
            elif vtype == 'entailment':
                prompt += """
        - **Entailment Queries**: List 2-3 logically implied questions
        """
        
        prompt += """
        
        ## 2. MULTI-PATH EXPLORATION
        Identify different interpretation paths for ambiguous queries:
        - Technical vs. General interpretations
        - Commercial vs. Informational intents
        - Different user contexts (beginner vs. expert)
        
        ## 3. CONTENT ARCHITECTURE
        Based on the fan-out analysis, provide:
        - **Primary Content Hub**: Main pillar page structure
        - **Supporting Content**: List of supporting articles needed
        - **Content Depth**: Word count recommendations for each piece
        - **Internal Linking Strategy**: How to connect the content
        """
        
        if settings.get('ai_search_type') in ['ai_overviews', 'both']:
            prompt += """
        
        ## 4. AI OVERVIEWS OPTIMIZATION
        For quick answer optimization:
        - **Direct Answer Format**: Exact 40-60 word answers for each query
        - **Snippet Structure**: Paragraph vs. list vs. table recommendations
        - **First 100 Words**: Optimization strategy for immediate visibility
        - **FAQ Structure**: Questions and concise answers
        """
        
        if settings.get('ai_search_type') in ['ai_mode', 'both']:
            prompt += """
        
        ## 5. AI MODE OPTIMIZATION
        For complex query fan-out:
        - **Passage-Level Coverage**: Key passages to include
        - **Semantic Completeness**: Topics that must be covered
        - **Entity Relationships**: Core entities and their connections
        - **Progressive Disclosure**: Information architecture strategy
        """
        
        if settings.get('include_entity_mapping'):
            prompt += """
        
        ## 6. ENTITY MAPPING
        - **Core Entities**: Primary entities to define
        - **Entity Relationships**: How entities connect
        - **Knowledge Graph**: Visual representation of connections
        - **Semantic Markup**: Schema.org recommendations
        """
        
        if settings.get('include_cross_verification'):
            prompt += """
        
        ## 7. CROSS-VERIFICATION STRATEGY
        - **Fact Verification**: Key facts to verify and cite
        - **Contradictory Information**: How to handle conflicting data
        - **Authority Signals**: Sources and citations to include
        - **Trust Indicators**: Elements that build credibility
        """
        
        if settings.get('include_schema'):
            prompt += """
        
        ## 8. SCHEMA MARKUP STRATEGY
        - **Essential Schemas**: Required schema types
        - **FAQ Schema**: Questions and answers
        - **HowTo Schema**: Step-by-step processes
        - **Article/BlogPosting**: Metadata requirements
        """
        
        if settings.get('include_competitors'):
            prompt += """
        
        ## 9. COMPETITIVE DIFFERENTIATION
        - **Content Gaps**: What competitors likely miss
        - **Unique Angles**: Fresh perspectives to explore
        - **10x Content**: How to create superior content
        - **Differentiation Strategy**: Unique value propositions
        """
        
        prompt += """
        
        ## 10. IMPLEMENTATION ROADMAP
        Provide a prioritized action plan:
        1. **Quick Wins**: Content that can rank quickly
        2. **Foundation Content**: Essential pieces to create first
        3. **Supporting Content**: Secondary pieces to develop
        4. **Enhancement Strategy**: Ongoing optimization approach
        
        ## 11. SUCCESS METRICS
        - **Ranking Targets**: Expected positions for each query
        - **Visibility Indicators**: AI mode appearance signals
        - **Engagement Metrics**: User behavior targets
        - **Conversion Goals**: Business outcomes to track
        
        Format your response with clear sections, bullet points, and actionable recommendations.
        Focus on practical implementation using Google's Query Fan-Out methodology.
        """
        
        return prompt


class ContentAnalyzer:
    """Handle content fetching and analysis for existing pages"""
    
    @staticmethod
    def fetch_content(url):
        """Fetch and parse content from a URL"""
        try:
            headers = {'User-Agent': Config.USER_AGENT}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract content
            content_data = {
                'url': url,
                'title': ContentAnalyzer._extract_title(soup),
                'meta_description': ContentAnalyzer._extract_meta_description(soup),
                'headings': ContentAnalyzer._extract_headings(soup),
                'content': ContentAnalyzer._extract_text_content(soup),
                'images': ContentAnalyzer._extract_images(soup),
                'internal_links': ContentAnalyzer._extract_internal_links(soup, url),
                'external_links': ContentAnalyzer._extract_external_links(soup, url),
                'structured_data': ContentAnalyzer._extract_structured_data(soup),
                'word_count': 0,
                'fetch_time': datetime.now()
            }
            
            # Calculate word count
            if content_data['content']:
                content_data['word_count'] = len(content_data['content'].split())
            
            return content_data
            
        except requests.RequestException as e:
            st.error(f"Error fetching content: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error parsing content: {str(e)}")
            return None
    
    @staticmethod
    def _extract_title(soup):
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return "No title found"
    
    @staticmethod
    def _extract_meta_description(soup):
        """Extract meta description"""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            return meta_desc.get('content', '').strip()
        return ""
    
    @staticmethod
    def _extract_headings(soup):
        """Extract all headings with hierarchy"""
        headings = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            headings.append({
                'level': tag.name,
                'text': tag.get_text().strip()
            })
        return headings
    
    @staticmethod
    def _extract_text_content(soup):
        """Extract main text content"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find main content area
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '.content', '#content']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body')
        
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
            # Clean up excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            return text
        
        return ""
    
    @staticmethod
    def _extract_images(soup):
        """Extract image information"""
        images = []
        for img in soup.find_all('img'):
            images.append({
                'src': img.get('src', ''),
                'alt': img.get('alt', ''),
                'title': img.get('title', '')
            })
        return images
    
    @staticmethod
    def _extract_internal_links(soup, base_url):
        """Extract internal links"""
        internal_links = []
        domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if domain in href or href.startswith('/'):
                internal_links.append({
                    'url': href,
                    'text': link.get_text().strip()
                })
        
        return internal_links
    
    @staticmethod
    def _extract_external_links(soup, base_url):
        """Extract external links"""
        external_links = []
        domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http') and domain not in href:
                external_links.append({
                    'url': href,
                    'text': link.get_text().strip()
                })
        
        return external_links
    
    @staticmethod
    def _extract_structured_data(soup):
        """Extract structured data (JSON-LD)"""
        structured_data = []
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                import json
                data = json.loads(script.string)
                structured_data.append(data)
            except:
                pass
        return structured_data
    
    @staticmethod
    def analyze_existing_content(content_data, primary_keyword, additional_keywords, 
                                competitor_urls, api_key, analysis_settings):
        """
        Analyze existing content using Query Fan-Out methodology
        """
        if not api_key:
            st.error("Please provide a Gemini API key")
            return None
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        model_name = analysis_settings.get('gemini_model', 'gemini-1.5-flash')
        model = genai.GenerativeModel(model_name)
        
        # Fetch competitor content if provided
        competitor_data = []
        if competitor_urls and analysis_settings.get('include_competitors'):
            for comp_url in competitor_urls[:3]:  # Limit to 3 competitors
                comp_content = ContentAnalyzer.fetch_content(comp_url)
                if comp_content:
                    competitor_data.append(comp_content)
        
        # Build the analysis prompt
        prompt = ContentAnalyzer._build_optimization_prompt(
            content_data, 
            primary_keyword, 
            additional_keywords,
            competitor_data,
            analysis_settings
        )
        
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error during Gemini analysis: {str(e)}")
            return None
    
    @staticmethod
    def _build_optimization_prompt(content_data, primary_keyword, additional_keywords, 
                                  competitor_data, settings):
        """Build prompt for existing content optimization"""
        
        # Prepare content summary
        content_summary = f"""
        URL: {content_data['url']}
        Title: {content_data['title']}
        Meta Description: {content_data['meta_description']}
        Word Count: {content_data['word_count']}
        Headings Count: {len(content_data['headings'])}
        Images: {len(content_data['images'])}
        Internal Links: {len(content_data['internal_links'])}
        External Links: {len(content_data['external_links'])}
        """
        
        # Heading structure
        heading_structure = "\n".join([f"{h['level'].upper()}: {h['text']}" 
                                      for h in content_data['headings'][:20]])
        
        # Content excerpt (first 500 words)
        content_excerpt = ' '.join(content_data['content'].split()[:500])
        
        prompt = f"""
        You are an expert in Google's Query Fan-Out system and content optimization.
        
        TASK: Analyze and provide optimization recommendations for existing content.
        
        PRIMARY KEYWORD: {primary_keyword}
        ADDITIONAL KEYWORDS: {', '.join(additional_keywords) if additional_keywords else 'None'}
        
        CURRENT CONTENT ANALYSIS:
        {content_summary}
        
        HEADING STRUCTURE:
        {heading_structure}
        
        CONTENT EXCERPT (First 500 words):
        {content_excerpt}
        
        OPTIMIZATION TARGET: {settings.get('ai_search_type', 'ai_mode').replace('_', ' ').title()}
        
        Please provide a comprehensive optimization analysis:
        
        ## 1. CURRENT STATE ASSESSMENT
        - **Content Coverage**: How well does the content cover the topic?
        - **Query Alignment**: Does it match user search intent?
        - **Semantic Completeness**: Missing topics or subtopics
        - **Technical Issues**: SEO problems identified
        
        ## 2. QUERY FAN-OUT ANALYSIS
        Based on the primary keyword "{primary_keyword}", generate:
        """
        
        # Add variant type analysis
        for vtype in settings.get('variant_types', ['equivalent', 'follow_up']):
            if vtype == 'equivalent':
                prompt += """
        - **Equivalent Queries**: Alternative queries this content should target
        """
            elif vtype == 'follow_up':
                prompt += """
        - **Follow-up Queries**: Next questions users would ask
        """
            elif vtype == 'specification':
                prompt += """
        - **Specification Queries**: Detailed queries to cover
        """
        
        prompt += """
        
        ## 3. CONTENT GAPS & OPPORTUNITIES
        - **Missing Query Coverage**: Which fan-out queries aren't addressed?
        - **Thin Content Areas**: Sections that need expansion
        - **New Sections Needed**: Additional content to add
        - **Entity Gaps**: Important entities not mentioned
        """
        
        if settings.get('analyze_structure'):
            prompt += """
        
        ## 4. STRUCTURAL OPTIMIZATION
        - **Heading Hierarchy**: Improvements to H1-H6 structure
        - **Content Flow**: Logical progression recommendations
        - **Paragraph Optimization**: Length and readability
        - **List Opportunities**: Where to use bullets/numbers
        """
        
        if settings.get('analyze_readability'):
            prompt += """
        
        ## 5. READABILITY & USER EXPERIENCE
        - **Sentence Complexity**: Simplification opportunities
        - **Technical Jargon**: Terms to explain or simplify
        - **Scannability**: Formatting improvements
        - **Engagement Elements**: Interactive elements to add
        """
        
        if settings.get('ai_search_type') in ['ai_overviews', 'both']:
            prompt += """
        
        ## 6. AI OVERVIEWS OPTIMIZATION
        - **Direct Answer**: Add a 40-60 word answer at the top
        - **Featured Snippet**: Format for snippet extraction
        - **FAQ Section**: Questions and answers to add
        - **Quick Reference**: Summary boxes or tables
        """
        
        if settings.get('ai_search_type') in ['ai_mode', 'both']:
            prompt += """
        
        ## 7. AI MODE OPTIMIZATION
        - **Passage Enhancement**: Key passages to improve
        - **Semantic Coverage**: Topics to add for completeness
        - **Entity Markup**: Entities to define and link
        - **Context Layers**: Progressive disclosure improvements
        """
        
        if settings.get('include_entity_mapping'):
            prompt += """
        
        ## 8. ENTITY OPTIMIZATION
        - **Missing Entities**: Important entities to add
        - **Entity Definitions**: Terms that need explanation
        - **Relationship Mapping**: Connections to establish
        - **Knowledge Graph**: Visual representation suggestions
        """
        
        if competitor_data:
            prompt += f"""
        
        ## 9. COMPETITIVE ANALYSIS
        Comparing to {len(competitor_data)} competitor(s):
        - **Content Length**: How does word count compare?
        - **Topic Coverage**: What do competitors cover that you don't?
        - **Unique Value**: What unique value can you add?
        - **Differentiation**: How to stand out from competition
        """
        
        if settings.get('include_schema'):
            prompt += """
        
        ## 10. SCHEMA MARKUP RECOMMENDATIONS
        - **Current Schema**: Analysis of existing structured data
        - **Missing Schema**: Types to add
        - **Schema Enhancements**: Properties to include
        - **Implementation Priority**: Which schemas are most important
        """
        
        prompt += """
        
        ## 11. ACTIONABLE OPTIMIZATION PLAN
        Provide specific, implementable recommendations:
        
        ### IMMEDIATE FIXES (Quick Wins)
        - Title tag optimization
        - Meta description rewrite
        - First paragraph enhancement
        - Quick formatting fixes
        
        ### SHORT-TERM IMPROVEMENTS (1-2 weeks)
        - Content additions (specify exact sections)
        - Heading restructuring
        - Internal linking improvements
        - Image optimization
        
        ### LONG-TERM ENHANCEMENTS (1 month)
        - Major content expansions
        - New supporting content creation
        - Comprehensive entity coverage
        - Advanced schema implementation
        
        ## 12. CONTENT REWRITE EXAMPLES
        Provide specific rewrite examples for:
        - Opening paragraph (optimized for AI)
        - Key sections that need improvement
        - FAQ additions
        - Conclusion with clear CTAs
        
        ## 13. SUCCESS METRICS
        - Expected ranking improvements
        - AI visibility indicators to monitor
        - User engagement targets
        - Conversion optimization goals
        
        Be specific, actionable, and prioritize recommendations by impact.
        Focus on practical changes that align with Google's Query Fan-Out system.
        """
        
        return prompt


class UIHelpers:
    """Helper functions for Streamlit UI"""
    
    @staticmethod
    def display_content_metrics(content_data):
        """Display content metrics in a nice format"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Word Count", f"{content_data['word_count']:,}")
        
        with col2:
            st.metric("Headings", len(content_data['headings']))
        
        with col3:
            st.metric("Images", len(content_data['images']))
        
        with col4:
            st.metric("Internal Links", len(content_data['internal_links']))
        
        # Show additional details in expander
        with st.expander("📊 Detailed Content Analysis"):
            st.write(f"**Title:** {content_data['title']}")
            st.write(f"**Meta Description:** {content_data['meta_description'] or 'Not found'}")
            
            if content_data['headings']:
                st.write("**Heading Structure:**")
                for h in content_data['headings'][:10]:
                    indent = "  " * (int(h['level'][1]) - 1)
                    st.write(f"{indent}{h['level'].upper()}: {h['text']}")
                if len(content_data['headings']) > 10:
                    st.write(f"... and {len(content_data['headings']) - 10} more headings")
            
            if content_data['structured_data']:
                st.write(f"**Structured Data Found:** {len(content_data['structured_data'])} schema(s)")
    
    @staticmethod
    def show_export_options(analysis, data, settings, mode='new_content'):
        """Show export options for the analysis"""
        col1, col2, col3 = st.columns(3)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with col1:
            # Markdown export
            st.download_button(
                label="📥 Download Analysis (Markdown)",
                data=analysis,
                file_name=f"query_fanout_{mode}_{timestamp}.md",
                mime="text/markdown"
            )
        
        with col2:
            # Full report
            if mode == 'new_content':
                report = f"""# Query Fan-Out Analysis Report - New Content Planning
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Target Queries
{chr(10).join(f"- {q}" for q in data[:settings.get('max_queries', 20)])}

## Analysis Settings
- Optimization Target: {settings.get('ai_search_type', 'ai_mode')}
- Depth: {settings.get('depth', 'Standard')}
- Variant Types: {', '.join(settings.get('variant_types', []))}

## Analysis Results
{analysis}

---
*Generated by Query Fan-Out Analysis Tool*
"""
            else:
                report = f"""# Query Fan-Out Analysis Report - Content Optimization
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Content Details
- URL: {data.get('url')}
- Primary Keyword: {data.get('keyword')}

## Analysis Settings
- Optimization Target: {settings.get('ai_search_type', 'ai_mode')}
- Depth: {settings.get('depth', 'Standard')}

## Optimization Recommendations
{analysis}

---
*Generated by Query Fan-Out Analysis Tool*
"""
            
            st.download_button(
                label="📄 Download Full Report",
                data=report,
                file_name=f"query_fanout_report_{timestamp}.md",
                mime="text/markdown"
            )
        
        with col3:
            # JSON export
            import json
            json_data = {
                'mode': mode,
                'timestamp': datetime.now().isoformat(),
                'data': data if isinstance(data, dict) else {'queries': data},
                'settings': settings,
                'analysis': analysis
            }
            
            st.download_button(
                label="💾 Download JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"query_fanout_data_{timestamp}.json",
                mime="application/json"
            )
