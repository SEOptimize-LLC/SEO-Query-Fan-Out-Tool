# Query Fan-Out SEO Analyzer

A comprehensive Streamlit application for analyzing and optimizing content using Google's Query Fan-Out technique for improved SEO performance in AI-powered search results.

## 🎯 Features

- **Sitemap Processing**: Automatically fetch and analyze all URLs from XML sitemaps
- **Batch Analysis**: Process up to 1,000+ URLs efficiently with rate limiting
- **Content Analysis**: 
  - Word count and structure analysis
  - Heading hierarchy evaluation
  - Schema markup detection
  - Readability scoring
- **Query Prediction**: Generate potential sub-queries based on content
- **Google Search Console Integration**: Fetch real query data and performance metrics
- **Semantic Clustering**: Group similar content for topical authority analysis
- **Actionable Insights**: Get specific recommendations for content optimization
- **Export Options**: Download results in CSV, JSON, or Excel formats

## 📋 Prerequisites

- Python 3.8 or higher
- Google Search Console API access (optional but recommended)
- Sufficient RAM (4-8GB recommended for processing 1,000 URLs)

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/query-fanout-analyzer.git
cd query-fanout-analyzer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Copy the `.env.example` file to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```
GOOGLE_API_KEY=your_api_key_here
GSC_CREDENTIALS_PATH=path/to/credentials.json
```

### 4. Download spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5. Run the application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🔧 Configuration

### Google Search Console Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select existing
3. Enable the Search Console API
4. Create a service account and download credentials
5. Add the service account email to your GSC property with "Read" permissions
6. Place the credentials JSON file in your project and update `.env`

### Advanced Settings

Edit `config.py` to customize:
- Batch sizes and rate limits
- Content analysis thresholds
- Clustering parameters
- Export settings

## 📊 Usage Guide

### 1. Input Methods

**Sitemap URL** (Recommended for full site analysis):
```
https://example.com/sitemap.xml
```

**CSV Upload**: 
- Must contain a column named 'url' or 'URL'
- Additional columns will be preserved

**Manual URLs**:
- Enter one URL per line
- Useful for testing specific pages

### 2. Analysis Process

1. **Content Fetching**: Downloads and parses HTML content
2. **Structure Analysis**: Evaluates headings, paragraphs, lists
3. **Query Prediction**: Generates potential search queries
4. **Coverage Scoring**: Calculates optimization score (0-1)
5. **GSC Integration**: Fetches real search performance data

### 3. Understanding Results

**Coverage Score Components**:
- Content length (20%)
- Structure quality (25%)
- Technical SEO (15%)
- Readability (20%)
- Semantic coverage (20%)

**Key Metrics**:
- **Word Count**: Aim for 1,000-1,500 minimum
- **H2 Count**: At least 5 for good structure
- **Schema Markup**: Required for rich snippets
- **Query Diversity**: More unique queries = better coverage

### 4. Taking Action

**High Priority Issues**:
- Pages with coverage score < 0.5
- Content under 500 words
- Missing schema markup
- Poor heading structure

**Optimization Steps**:
1. Expand thin content
2. Add relevant subheadings
3. Implement schema markup
4. Improve meta descriptions
5. Target query variations

## 🏗️ Architecture

```
query-fanout-analyzer/
├── app.py                 # Main Streamlit application
├── utils.py              # Utility functions and classes
├── gsc_integration.py    # Google Search Console integration
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md            # This file
```

## 🚀 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Set environment variables in Streamlit settings
4. Deploy!

### Local Deployment

For production use:

```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn app:server --bind 0.0.0.0:8000
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📈 Performance Optimization

### For Large-Scale Analysis

1. **Batch Processing**: Adjust batch size based on server capacity
2. **Caching**: Enable Redis for improved performance
3. **Async Processing**: Built-in async support for concurrent requests
4. **Rate Limiting**: Automatic throttling to respect API limits

### Memory Management

- Processes URLs in chunks
- Clears memory between batches
- Efficient data structures for large datasets

## 🐛 Troubleshooting

### Common Issues

**"Rate limit exceeded"**
- Reduce batch size in settings
- Add delays between requests
- Check API quotas

**"Memory error"**
- Process fewer URLs at once
- Increase server RAM
- Enable result streaming

**"Connection timeout"**
- Check internet connectivity
- Verify URL accessibility
- Increase timeout in config

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on Google's Query Fan-Out patent and research
- Inspired by the SEO community's work on AI search optimization
- Built with Streamlit for easy deployment and usage

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review the troubleshooting guide

---

**Note**: This tool is for SEO analysis and optimization. Always follow Google's webmaster guidelines and focus on creating high-quality, user-first content.
