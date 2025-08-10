"""
Utility functions for Query Fan-Out Analysis
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import re
from urllib.parse import urlparse
import json

# Optional imports with error handling
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    st.warning("Web scraping libraries not installed. Please run: pip install requests beautifulsoup4")

from config import Config


class APIValidator:
    """Validate API keys for different providers"""
    
    @staticmethod
    def validate_gemini_key(api_key):
        """Validate Gemini API key"""
        if not api_key:
            return False, "No API key provided"
        
        # Check basic format
        if not api_key.startswith('AIza'):
            return False, "Invalid key format. Gemini keys should start with 'AIza'"
        
        if len(api_key) < 30:
            return False, "Key appears to be incomplete"
        
        # Try to make a simple API call
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # Try to list models (lightweight call)
            models = genai.list_models()
            model_list = list(models)
            
            return True, f"✅ Valid key! Access to {len(model_list)} models confirmed"
        except Exception as e:
            error_str = str(e)
            
            if "API_KEY_INVALID" in error_str:
                return False, "❌ Invalid API key. Please check your key at https://makersuite.google.com/app/apikey"
            elif "PERMISSION_DENIED" in error_str:
                return False, "❌ API key valid but lacks permissions. Enable the Generative Language API in Google Cloud Console"
            elif "quota" in error_str.lower():
                return False, "❌ API quota exceeded. Check your usage limits"
            else:
                return False, f"❌ Validation failed: {error_str[:100]}"
    
    @staticmethod
    def validate_openai_key(api_key):
        """Validate OpenAI API key"""
        if not api_key:
            return False, "No API key provided"
        
        if not api_key.startswith('sk-'):
            return False, "Invalid key format. OpenAI keys should start with 'sk-'"
        
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            # Try to list models
            models = client.models.list()
            model_count = len(list(models))
            
            return True, f"✅ Valid key! Access to OpenAI models confirmed"
        except Exception as e:
            error_str = str(e)
            
            if "incorrect api key" in error_str.lower() or "invalid" in error_str.lower():
                return False, "❌ Invalid API key. Get a new one at https://platform.openai.com/api-keys"
            elif "quota" in error_str.lower():
                return False, "❌ API quota exceeded or no credits available"
            else:
                return False, f"❌ Validation failed: {error_str[:100]}"
    
    @staticmethod
    def validate_anthropic_key(api_key):
        """Validate Anthropic API key"""
        if not api_key:
            return False, "No API key provided"
        
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            
            # Try a minimal API call
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            
            return True, "✅ Valid key! Claude API access confirmed"
        except Exception as e:
            error_str = str(e)
            
            if "invalid" in error_str.lower() or "unauthorized" in error_str.lower():
                return False, "❌ Invalid API key. Get a new one at https://console.anthropic.com/settings/keys"
            elif "credit" in error_str.lower() or "balance" in error_str.lower():
                return False, "❌ No credits available. Add credits to your Anthropic account"
            else:
                return False, f"❌ Validation failed: {error_str[:100]}"


class AIProvider:
    """Handle different AI provider integrations"""
    
    @staticmethod
    def generate_content(prompt, api_key, settings):
        """Generate content using the selected AI provider"""
        provider = settings.get('ai_provider', 'gemini')
        model = settings.get('model', 'gemini-1.5-flash')
        
        if provider == 'gemini':
            return AIProvider._generate_gemini(prompt, api_key, model)
        elif provider == 'openai':
            return AIProvider._generate_openai(prompt, api_key, model)
        elif provider == 'anthropic':
            return AIProvider._generate_anthropic(prompt, api_key, model)
        else:
            st.error(f"Unknown AI provider: {provider}")
            return None
    
    @staticmethod
    def _generate_gemini(prompt, api_key, model):
        """Generate content using Google Gemini"""
        if not GEMINI_AVAILABLE:
            st.error("Google Generative AI not installed. Run: pip install google-generativeai")
            return None
        
        try:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(model)
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Gemini API error: {str(e)}")
            return None
    
    @staticmethod
    def _generate_openai(prompt, api_key, model):
        """Generate content using OpenAI"""
        if not OPENAI_AVAILABLE:
            st.error("OpenAI library not installed. Run: pip install openai")
            return None
        
        try:
            client = openai.OpenAI(api_key=api_key)
            
            # Special handling for O1 models (reasoning models)
            if 'o1' in model:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=1,  # O1 models work best with temperature=1
                    max_completion_tokens=32000  # O1 models support more tokens
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert in Google's Query Fan-Out system and SEO optimization."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
            
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"OpenAI API error: {str(e)}")
            if "api_key" in str(e).lower():
                st.error("Please check your OpenAI API key")
            return None
    
    @staticmethod
    def _generate_anthropic(prompt, api_key, model):
        """Generate content using Anthropic Claude"""
        if not ANTHROPIC_AVAILABLE:
            st.error("Anthropic library not installed. Run: pip install anthropic")
            return None
        
        try:
            client = anthropic.Anthropic(api_key=api_key)
            
            response = client.messages.create(
                model=model,
                max_tokens=4000,
                temperature=0.7,
                system="You are an expert in Google's Query Fan-Out system and SEO optimization. Provide detailed, actionable recommendations.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract text from Claude's response
            return response.content[0].text
        except Exception as e:
            st.error(f"Anthropic API error: {str(e)}")
            if "api_key" in str(e).lower():
                st.error("Please check your Anthropic API key")
            return None


class QueryAnalyzer:
    """Handle query analysis and fan-out predictions"""
    
    @staticmethod
    def analyze_query_fanout_new_content(queries_df, api_key, analysis_settings):
        """
        Perform Query Fan-Out analysis for new content planning
        """
        if not api_key:
            st.error("Please provide an API key")
            return None
        
        # Build the analysis prompt
        queries_list = queries_df['query'].tolist()
        prompt = QueryAnalyzer._build_new_content_prompt(queries_list, analysis_settings)
        
        # Generate content using selected provider
        return AIProvider.generate_content(prompt, api_key, analysis_settings)
    
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
            elif vtype == 'canonicalization':
                prompt += """
        - **Canonicalization Queries**: List 2-3 standardized versions
        """
            elif vtype == 'clarification':
                prompt += """
        - **Clarification Queries**: List 2-3 questions to clarify intent
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
        if not SCRAPING_AVAILABLE:
            st.error("Web scraping libraries are not installed. Please install requests and beautifulsoup4")
            return None
            
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
            st.error("Please provide an API key")
            return None
        
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
        
        # Generate content using selected provider
        return AIProvider.generate_content(prompt, api_key, analysis_settings)
    
    @staticmethod
    def _build_optimization_prompt(content_data, primary_keyword, additional_keywords, 
                                  competitor_data, settings):
        """Build prompt for existing content optimization with actionable rewrites"""
        
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
        
        # Content excerpt (first 1000 words for better context)
        content_excerpt = ' '.join(content_data['content'].split()[:1000])
        
        prompt = f"""
        You are an expert content optimizer specializing in Google's Query Fan-Out system and AI Overviews optimization.
        
        CRITICAL INSTRUCTION: Provide SPECIFIC, ACTIONABLE REWRITES that can be directly copy-pasted into the content. Not generic advice.
        
        TARGET OPTIMIZATION:
        - Primary Keyword: {primary_keyword}
        - Additional Keywords: {', '.join(additional_keywords) if additional_keywords else 'None'}
        - Optimization Type: {settings.get('ai_search_type', 'ai_mode').replace('_', ' ').title()}
        
        CURRENT CONTENT ANALYSIS:
        {content_summary}
        
        HEADING STRUCTURE:
        {heading_structure}
        
        CONTENT SAMPLE (First 1000 words):
        {content_excerpt}
        
        Provide the following ACTIONABLE optimization report:
        
        # 📊 QUERY FAN-OUT OPTIMIZATION REPORT
        
        ## 1. 🎯 QUICK WIN: ANSWER BOX OPTIMIZATION
        
        ### Current Opening (if exists)
        ```
        [Extract and show the current opening paragraph]
        ```
        
        ### OPTIMIZED REWRITE FOR AI OVERVIEW
        ```
        [Provide a completely rewritten opening that:
        - Answers the primary query in the first 40-60 words
        - Uses natural, conversational tone matching the original style
        - Includes the primary keyword naturally
        - Structures for featured snippet extraction]
        ```
        
        ### Implementation Notes
        - Why this works better: [Brief explanation]
        - Expected impact: [Specific ranking/CTR improvement]
        
        ## 2. 🔍 QUERY VARIANT COVERAGE ANALYSIS
        
        Based on "{primary_keyword}", these query variants MUST be addressed:
        
        ### Missing Query Variants Your Content Doesn't Answer:
        1. **[Specific query variant]** - Not covered
        2. **[Specific query variant]** - Partially covered in [section]
        3. **[Specific query variant]** - Not covered
        
        ### NEW SECTIONS TO ADD (Copy-Paste Ready)
        
        #### New Section 1: [Specific Heading]
        ```markdown
        ## [Exact Heading to Add]
        
        [Write 150-200 words of ready-to-use content that:
        - Answers the specific query variant
        - Maintains the original writing style
        - Includes relevant entities and keywords
        - Uses proper formatting for AI extraction]
        ```
        
        #### New Section 2: [Specific Heading]
        ```markdown
        ## [Exact Heading to Add]
        
        [Write another 150-200 words of ready-to-use content]
        ```
        
        ## 3. 📝 CRITICAL CONTENT REWRITES
        
        ### REWRITE 1: [Specific Section Name]
        
        **Current Version Issues:**
        - [Specific problem 1]
        - [Specific problem 2]
        
        **OPTIMIZED REWRITE:**
        ```markdown
        [Provide complete rewritten section that:
        - Fixes the identified issues
        - Adds query variant coverage
        - Improves for AI extraction
        - Maintains original tone and style]
        ```
        
        ### REWRITE 2: [Another Section Name]
        
        **Current Version Issues:**
        - [Specific problem]
        
        **OPTIMIZED REWRITE:**
        ```markdown
        [Complete rewritten section]
        ```
        
        ## 4. 🏗️ STRUCTURAL IMPROVEMENTS
        
        ### Current Structure Problem
        [Identify specific structural issue]
        
        ### EXACT RESTRUCTURING PLAN
        ```
        1. Move [specific section] to position 2
        2. Combine [section X] with [section Y] under new heading: "[New Heading]"
        3. Break up [long section] into:
           - [New subsection 1]
           - [New subsection 2]
           - [New subsection 3]
        4. Add new section after [specific location]: "[New Section Title]"
        ```
        
        ## 5. 🎨 METADATA OPTIMIZATION
        
        ### Current Title Tag
        ```
        {content_data['title']}
        ```
        
        ### OPTIMIZED Title Tag
        ```
        [New title that's exactly 50-60 characters, includes primary keyword, and encourages clicks]
        ```
        
        ### Current Meta Description
        ```
        {content_data['meta_description']}
        ```
        
        ### OPTIMIZED Meta Description
        ```
        [New description that's exactly 150-160 characters, includes primary keyword, has clear CTA]
        ```
        
        ## 6. 📋 FAQ SECTION TO ADD (Copy-Paste Ready)
        
        Add this exact FAQ section to capture more query variants:
        
        ```markdown
        ## Frequently Asked Questions
        
        ### [Question 1 that targets a specific query variant]
        [40-60 word answer that's voice-search optimized]
        
        ### [Question 2 that targets another variant]
        [40-60 word answer]
        
        ### [Question 3 that targets another variant]
        [40-60 word answer]
        
        ### [Question 4 that targets another variant]
        [40-60 word answer]
        ```
        
        ## 7. 🔗 INTERNAL LINKING OPTIMIZATION
        
        ### Links to ADD:
        1. In paragraph about [topic], add link to [URL] with anchor text "[exact anchor text]"
        2. In section [X], add link to [URL] with anchor text "[exact anchor text]"
        3. After [specific sentence], add: "Learn more about [anchor text](URL)"
        
        ### Links to REMOVE or UPDATE:
        1. Remove link in [location] - reason: [specific reason]
        2. Change "[current anchor]" to "[new anchor]" in [location]
        
        ## 8. 📊 SCHEMA MARKUP TO IMPLEMENT
        
        ### FAQ Schema (Copy-Paste Ready)
        ```json
        {
          "@context": "https://schema.org",
          "@type": "FAQPage",
          "mainEntity": [
            {
              "@type": "Question",
              "name": "[Question 1]",
              "acceptedAnswer": {
                "@type": "Answer",
                "text": "[Answer 1]"
              }
            }
          ]
        }
        ```
        
        ## 9. 🚀 IMPLEMENTATION PRIORITY
        
        ### DO TODAY (15 minutes):
        1. [ ] Copy-paste the new opening paragraph
        2. [ ] Update title tag and meta description
        3. [ ] Add the FAQ section at the end
        
        ### DO THIS WEEK (1-2 hours):
        1. [ ] Implement all content rewrites from Section 3
        2. [ ] Add the new sections from Section 2
        3. [ ] Restructure content per Section 4
        
        ### DO THIS MONTH:
        1. [ ] Implement schema markup
        2. [ ] Update internal linking
        3. [ ] Create supporting content for query variants
        
        ## 10. 📈 EXPECTED RESULTS
        
        After implementing these SPECIFIC changes:
        - **Week 1**: Expect improved CTR (+15-20%) from better meta tags
        - **Week 2-4**: See featured snippet appearance for 2-3 query variants
        - **Month 2**: Achieve page 1 rankings for additional 3-5 long-tail variants
        - **Month 3**: Potential AI Overview inclusion for primary keyword
        
        ## 11. 🎯 COMPETITOR GAP ANALYSIS
        
        """
        
        if competitor_data:
            prompt += """
        Based on competitor analysis, here's what they have that you're missing:
        
        ### Content They Cover That You Don't:
        1. [Specific topic/section]
        2. [Specific topic/section]
        3. [Specific topic/section]
        
        ### ADD THIS SECTION to compete:
        ```markdown
        [Write ready-to-use content section that fills the gap]
        ```
        """
        
        prompt += """
        
        ---
        
        Remember: Every suggestion above is SPECIFIC and ACTIONABLE. The user should be able to copy-paste rewrites directly into their CMS without any additional editing needed.
        
        Focus on maintaining the original writing style and brand voice while optimizing for Query Fan-Out and AI Overviews.
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
AI Provider: {settings.get('ai_provider', 'Unknown')}
Model: {settings.get('model', 'Unknown')}

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
AI Provider: {settings.get('ai_provider', 'Unknown')}
Model: {settings.get('model', 'Unknown')}

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
            json_data = {
                'mode': mode,
                'timestamp': datetime.now().isoformat(),
                'ai_provider': settings.get('ai_provider'),
                'model': settings.get('model'),
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
