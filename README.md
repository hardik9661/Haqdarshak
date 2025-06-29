# 🔍 Haqdarshak Scheme Research Tool

An automated AI-powered tool for analyzing government schemes and extracting key information using LangChain, FAISS, and OpenAI/OpenRouter APIs.

## Features

- 🔍 **URL Processing**: Load and analyze scheme information from URLs or PDFs
- 🤖 **AI-Powered Analysis**: Generate comprehensive summaries covering:
  - Scheme Benefits
  - Scheme Application Process
  - Eligibility Criteria
  - Documents Required
- ❓ **Q&A Interface**: Ask specific questions about schemes and get detailed answers
- 📚 **Source References**: All answers include source URLs and content snippets
- 🔗 **OpenRouter Support**: Use OpenRouter API for cost-effective AI model access

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key

#### Option A: OpenRouter (Recommended)
1. Get a free API key from [OpenRouter](https://openrouter.ai/)
2. Edit the `.config` file and replace `your-openrouter-api-key-here` with your actual API key

#### Option B: OpenAI
1. Get an API key from [OpenAI](https://platform.openai.com/)
2. Edit the `.config` file and replace `your-openai-api-key-here` with your actual API key

### 3. Start the Application
```bash
streamlit run main.py
```

The application will be available at `http://localhost:8501`

## Usage

### 1. Input URLs
- **Direct Input**: Enter URLs directly in the sidebar
- **File Upload**: Upload a text file containing URLs (one per line)

### 2. Process Content
- Click "Process URLs" to load and analyze the scheme documents
- The system will extract content, create embeddings, and build a searchable index

### 3. Generate Summaries
- Click "Generate Scheme Summary" to get comprehensive analysis covering all four key criteria

### 4. Ask Questions
- Use the Q&A interface to ask specific questions about the schemes
- Get detailed answers with source references

## Sample URL
Test the application with this sample URL:
```
https://mohua.gov.in/upload/uploadfiles/files/PMSVANidhi%20Guideline_English.pdf
```

## Configuration

### API Models
You can configure different models in the `.config` file:

#### OpenRouter Models
- `openai/gpt-3.5-turbo` (default)
- `openai/gpt-4`
- `anthropic/claude-3-haiku`
- `anthropic/claude-3-sonnet`
- `google/gemini-pro`

#### OpenAI Models
- `gpt-3.5-turbo`
- `gpt-4`
- `gpt-4-turbo`

### Chunk Configuration
- `CHUNK_SIZE`: Size of text chunks for processing (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

## Technical Details

### Architecture
- **Streamlit**: Web application framework
- **LangChain**: Document processing and LLM integration
- **FAISS**: Vector similarity search and indexing
- **OpenAI/OpenRouter**: AI model APIs for embeddings and text generation
- **Unstructured**: Document loading from URLs and PDFs

### File Structure
```
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .config                # Configuration file
├── faiss_store_openai.pkl # FAISS index storage (created automatically)
└── README.md              # This file
```

## Why OpenRouter?

OpenRouter provides several advantages:
- **Cost-Effective**: Often cheaper than direct OpenAI API
- **Multiple Models**: Access to various AI models (OpenAI, Anthropic, Google, etc.)
- **Free Tier**: Generous free tier for testing
- **Easy Setup**: Simple API key setup

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing dependencies with `pip install -r requirements.txt`
2. **API Key Error**: Ensure your API key is correctly set in the `.config` file
3. **URL Loading Issues**: Some URLs may require authentication or have access restrictions

### Getting Help
- Check the console output for detailed error messages
- Ensure all dependencies are installed correctly
- Verify your API key is valid and has sufficient credits

## License

This project is developed for Haqdarshak's scheme research requirements.

## 🎯 Key Features

### Core Functionality
- **URL Processing**: Load and process scheme information from URLs or PDF files
- **Intelligent Summarization**: Generate comprehensive summaries covering four key criteria:
  - Scheme Benefits
  - Scheme Application Process
  - Eligibility Criteria
  - Documents Required
- **Q&A System**: Ask specific questions about schemes and get detailed answers
- **Source Tracking**: All answers include source references for transparency

### Technical Features
- **LangChain Integration**: Advanced document processing and LLM integration
- **FAISS Vector Search**: Fast and efficient similarity search for relevant information
- **OpenAI Embeddings**: High-quality vector embeddings for semantic search
- **Streamlit Interface**: User-friendly web application interface
- **Multi-format Support**: Handles both web pages and PDF documents

## 🛠️ Tech Stack

### Core Technologies
- **Streamlit** (1.28.1) - Web application framework
- **LangChain** (0.0.350) - Document processing and LLM orchestration
- **OpenAI** (1.3.7) - Embeddings and language model
- **FAISS** (1.7.4) - Vector similarity search
- **Unstructured** (0.11.8) - Document parsing and extraction

### Supporting Libraries
- **pandas** (2.1.3) - Data manipulation
- **numpy** (1.24.3) - Numerical operations
- **requests** (2.31.0) - HTTP requests
- **python-dotenv** (1.0.0) - Environment variable management
- **pypdf** (3.17.1) - PDF processing
- **beautifulsoup4** (4.12.2) - HTML parsing
- **tiktoken** (0.5.1) - Token counting

## 📁 Project Structure

```
Haqdarshak/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .config                # Configuration file (API keys, settings)
├── faiss_store_openai.pkl # FAISS index storage (generated)
└── README.md              # Project documentation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Internet connection for downloading packages

### Step 1: Clone/Download the Project
```bash
# If using git
git clone <repository-url>
cd Haqdarshak

# Or download and extract the project files
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure API Key
1. Open the `.config` file
2. Replace `your-openai-api-key-here` with your actual OpenAI API key
3. Save the file

```bash
# Example .config file content
OPENAI_API_KEY=sk-your-actual-api-key-here
EMBEDDING_MODEL=text-embedding-ada-002
LLM_MODEL=gpt-3.5-turbo
FAISS_INDEX_PATH=faiss_store_openai.pkl
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Step 4: Run the Application
```bash
streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Input URLs
- **Direct Input**: Enter URLs directly in the sidebar text area (one per line)
- **File Upload**: Upload a text file containing URLs (one per line)

### 2. Process Content
- Click the "🚀 Process URLs" button
- The system will:
  - Download and extract content from URLs/PDFs
  - Split documents into manageable chunks
  - Generate vector embeddings
  - Create FAISS index for fast retrieval

### 3. Generate Summaries
- Click "📋 Generate Scheme Summary" to create comprehensive summaries
- The tool will analyze content and provide structured information for:
  - Scheme Benefits
  - Application Process
  - Eligibility Criteria
  - Required Documents

### 4. Ask Questions
- Use the Q&A interface to ask specific questions about schemes
- Get detailed answers with source references
- View relevant document excerpts that support the answers

## 🔧 Configuration Options

### Environment Variables (.config file)
- `OPENAI_API_KEY`: Your OpenAI API key
- `EMBEDDING_MODEL`: OpenAI embedding model (default: text-embedding-ada-002)
- `LLM_MODEL`: OpenAI language model (default: gpt-3.5-turbo)
- `FAISS_INDEX_PATH`: Path for storing FAISS index
- `CHUNK_SIZE`: Document chunk size for processing (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

## 📊 Sample Usage

### Example URL
```
https://mohua.gov.in/upload/uploadfiles/files/PMSVANidhi%20Guideline_English.pdf
```

### Sample Questions
- "What are the eligibility criteria for this scheme?"
- "How much loan amount can be availed?"
- "What documents are required for application?"
- "What is the application process?"
- "What are the benefits of this scheme?"

## 🔍 How It Works

### 1. Document Processing
- **URL Loading**: Uses LangChain's UnstructuredURLLoader for web pages
- **PDF Processing**: Downloads and processes PDF files using PyPDFLoader
- **Text Extraction**: Extracts clean, structured text from various formats

### 2. Text Chunking
- **Recursive Splitting**: Uses RecursiveCharacterTextSplitter for intelligent text division
- **Configurable Chunks**: Adjustable chunk size and overlap for optimal processing
- **Metadata Preservation**: Maintains source information for each chunk

### 3. Vector Embeddings
- **OpenAI Embeddings**: Generates high-quality vector representations
- **Semantic Understanding**: Captures meaning and context of text chunks
- **Dimensionality**: Uses 1536-dimensional vectors for rich representation

### 4. FAISS Indexing
- **Fast Search**: FAISS provides sub-linear search complexity
- **Similarity Matching**: Finds most relevant chunks for queries
- **Persistent Storage**: Saves index to disk for reuse

### 5. Q&A System
- **Retrieval-Augmented Generation**: Combines retrieval with LLM generation
- **Context-Aware Answers**: Uses relevant document chunks as context
- **Source Attribution**: Provides source references for all answers

## 🎨 User Interface Features

### Modern Design
- **Responsive Layout**: Works on desktop and mobile devices
- **Intuitive Navigation**: Clear sidebar and main content areas
- **Visual Feedback**: Progress indicators and status messages
- **Expandable Sections**: Organized information display

### Interactive Elements
- **Real-time Processing**: Live updates during document processing
- **Expandable Sources**: Click to view source documents
- **Collapsible Summaries**: Organized summary sections
- **Error Handling**: Clear error messages and recovery options

## 🔒 Security & Privacy

### API Key Management
- **Environment Variables**: Secure storage of API keys
- **No Hardcoding**: Keys are never stored in source code
- **Local Processing**: All data processing happens locally

### Data Handling
- **Temporary Storage**: PDF files are processed and deleted
- **No Data Retention**: No personal data is stored permanently
- **Source Transparency**: All answers include source references

## 🐛 Troubleshooting

### Common Issues

#### 1. OpenAI API Key Error
```
Error: Please set your OpenAI API key in the .config file!
```
**Solution**: Update the `.config` file with your valid OpenAI API key

#### 2. Package Installation Issues
```
Error: Failed to install dependencies
```
**Solution**: 
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. PDF Processing Errors
```
Error: Failed to download PDF
```
**Solution**: Check URL accessibility and internet connection

#### 4. Memory Issues
```
Error: Out of memory during processing
```
**Solution**: Reduce chunk size in `.config` file or process fewer URLs at once

### Performance Optimization
- **Chunk Size**: Adjust `CHUNK_SIZE` in `.config` for optimal performance
- **Batch Processing**: Process URLs in smaller batches for large datasets
- **Model Selection**: Use appropriate OpenAI models based on requirements

## 📈 Future Enhancements

### Planned Features
- **Multi-language Support**: Support for regional languages
- **Batch Processing**: Process multiple schemes simultaneously
- **Export Functionality**: Export summaries to PDF/Word formats
- **Advanced Analytics**: Scheme comparison and trend analysis
- **User Authentication**: Multi-user support with role-based access

### Technical Improvements
- **Caching**: Implement intelligent caching for faster responses
- **Parallel Processing**: Multi-threaded document processing
- **Database Integration**: Persistent storage for processed schemes
- **API Endpoints**: RESTful API for external integrations

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include error handling
- Write clear commit messages

## 📄 License

This project is developed for Haqdarshak's internal use. Please refer to your organization's licensing policies.

## 📞 Support

For technical support or questions:
- Check the troubleshooting section above
- Review the configuration options
- Ensure all dependencies are properly installed
- Verify your OpenAI API key is valid and has sufficient credits

---

**Developed for Haqdarshak Research Team**  
*Empowering individuals with knowledge to access government schemes effectively* 
 #   H a q d a r s h a k  
 