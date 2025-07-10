# SEO Query Fan-Out Analysis Tool

A powerful Streamlit application that analyzes Google Search Console data using the Query Fan-Out methodology to optimize content for Google's AI Mode search.

## 🚀 Features

- **OAuth 2.0 Authentication**: Secure Google account authentication (no API keys needed from users)
- **Google Search Console Integration**: Direct access to your search query data
- **Query Fan-Out Analysis**: AI-powered analysis using Google's Gemini to predict query expansions
- **Content Gap Identification**: Discover missing content opportunities
- **AI Mode Optimization**: Recommendations specifically for Google's AI-powered search
- **Export Functionality**: Download analysis reports in Markdown format

## 🛠️ Setup Instructions

### 1. Google Cloud Console Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Google Search Console API**:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Google Search Console API"
   - Click "Enable"

### 2. Create OAuth 2.0 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client ID"
3. Configure the consent screen if prompted
4. Choose "Web application" as the application type
5. Add authorized redirect URIs:
   - For local testing: `http://localhost:8501`
   - For Streamlit Cloud: `https://your-app-name.streamlit.app`
6. Download the credentials JSON file

### 3. Get Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Save it securely

### 4. Deploy on Streamlit Cloud

1. Fork or clone this repository
2. Connect your GitHub repo to Streamlit Cloud
3. Add secrets in Streamlit Cloud dashboard:
   ```toml
   GOOGLE_CLIENT_ID = "your-client-id.apps.googleusercontent.com"
   GOOGLE_CLIENT_SECRET = "your-client-secret"
   REDIRECT_URI = "https://your-app-name.streamlit.app"
   GEMINI_API_KEY = "your-gemini-api-key"
   ```

### 5. Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/SEOptimize-LLC/SEO-Query-Fan-Out-Tool.git
   cd SEO-Query-Fan-Out-Tool
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create `.streamlit/secrets.toml` file with your credentials

4. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

## 📊 How It Works

1. **Authentication**: Users authenticate with their Google account
2. **Property Selection**: Choose from verified Google Search Console properties
3. **Data Fetching**: Retrieve search query data for analysis
4. **Query Fan-Out Analysis**: AI analyzes queries to predict:
   - Sub-queries Google AI might generate
   - Content gaps and opportunities
   - Semantic relationships between queries
5. **Recommendations**: Get actionable content optimization suggestions

## 🔍 Query Fan-Out Methodology

Query Fan-Out is Google's approach to AI-powered search where:
- A single query is expanded into multiple related sub-queries
- Content is evaluated at the passage level
- Semantic coverage matters more than keyword density

This tool helps you understand and optimize for this new search paradigm.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Google's Gemini AI](https://deepmind.google/technologies/gemini/)
- Inspired by the SEO community's research on AI Mode optimization

---

**Note**: This tool requires valid Google Search Console access and appropriate API credentials. Ensure you have the necessary permissions for the properties you want to analyze.
