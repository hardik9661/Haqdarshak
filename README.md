# 🔍 Haqdarshak Scheme Research Tool

An intelligent AI-powered web application for analyzing government schemes and extracting key information using advanced natural language processing and machine learning techniques.

<img width="937" alt="image" src="https://github.com/user-attachments/assets/b8b256b1-b453-4bd3-a042-efedd22c882b" />

## 🎯 Overview

The Haqdarshak Scheme Research Tool is designed to help researchers, government officials, and citizens understand government schemes more effectively. It automatically processes scheme documents from URLs or PDFs and provides comprehensive analysis covering key criteria such as benefits, application processes, eligibility, and required documents.

## ✨ Features

- **🔗 URL Processing**: Load and analyze scheme information from URLs or PDFs
- **🤖 AI-Powered Analysis**: Generate intelligent summaries using Hugging Face models
- **📋 Comprehensive Summaries**: Extract information covering four key criteria:
  - Scheme Benefits
  - Scheme Application Process
  - Eligibility Criteria
  - Documents Required
- **❓ Interactive Q&A**: Ask specific questions about schemes and get detailed answers
- **📚 Source References**: All answers include source URLs and content snippets
- **🔍 Vector Search**: Fast and accurate information retrieval using FAISS
- **📱 User-Friendly Interface**: Clean, responsive web interface built with Streamlit

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI/ML**: 
  - LangChain (Document processing and LLM integration)
  - Hugging Face (Embeddings and Language Models)
  - FAISS (Vector similarity search)
- **Document Processing**: 
  - Unstructured (URL and PDF content extraction)
  - PyPDF (PDF processing)
- **Language Models**: 
  - Sentence Transformers (Text embeddings)
  - Hugging Face Hub (LLM inference)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- Hugging Face API token

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/haqdarshak-scheme-research.git
   cd haqdarshak-scheme-research
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   - Get your Hugging Face API token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Edit the `.config` file and replace `your-huggingface-api-key` with your actual token

4. **Start the application**
   ```bash
   streamlit run main.py
   ```

5. **Access the application**
   - Open your browser and go to: `http://localhost:8501`

## 📖 Usage

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

## 📝 Sample Questions

You can ask various types of questions about government schemes:

### Scheme Benefits
- "What are the main benefits of this scheme?"
- "How does this scheme help beneficiaries?"
- "What financial support does the scheme provide?"

### Application Process
- "How can someone apply for this scheme?"
- "What is the step-by-step application process?"
- "Where can I submit my application?"

### Eligibility
- "Who is eligible for this scheme?"
- "What are the eligibility criteria?"
- "Are there any age or income restrictions?"

### Required Documents
- "What documents are required to apply?"
- "Do I need an Aadhaar card or income certificate?"
- "Is a caste certificate necessary?"

## 🔧 Configuration

### Model Configuration
You can configure different models in the `.config` file:

```ini
# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Language Model
LLM_MODEL=google/flan-t5-large

# Chunk Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Supported Models
- **Embedding Models**: `sentence-transformers/all-MiniLM-L6-v2`, `sentence-transformers/all-mpnet-base-v2`
- **Language Models**: `google/flan-t5-large`, `tiiuae/falcon-7b-instruct`, `mistralai/Mistral-7B-Instruct-v0.2`

## 📁 Project Structure

```
Haqdarshak/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .config                # Configuration file (API keys, models)
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── faiss_store_openai.pkl # FAISS index storage (auto-generated)
└── test_openrouter.py     # Test script for API verification
```

## 🔒 Security & Privacy

- **API Keys**: Store your API keys in the `.config` file (not committed to Git)
- **Data Processing**: All document processing happens locally
- **No Data Storage**: The application does not permanently store uploaded documents

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing dependencies with `pip install -r requirements.txt`
2. **API Key Error**: Ensure your Hugging Face API key is correctly set in the `.config` file
3. **Model Loading Issues**: Some models may take time to download on first use
4. **URL Loading Issues**: Some URLs may require authentication or have access restrictions

### Getting Help
- Check the console output for detailed error messages
- Ensure all dependencies are installed correctly
- Verify your API key is valid and has sufficient credits

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is developed for Haqdarshak's scheme research requirements.

## 🙏 Acknowledgments

- **Haqdarshak**: For the project requirements and domain expertise
- **Hugging Face**: For providing excellent open-source language models
- **LangChain**: For the powerful document processing and LLM integration framework
- **Streamlit**: For the intuitive web application framework

## 📞 Support:9818624454

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the error messages in the console
3. Ensure all dependencies and API keys are properly configured

---

**Built with ❤️ for better government scheme accessibility**
