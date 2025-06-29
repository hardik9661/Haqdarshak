import streamlit as st
import os
import pickle
import pandas as pd
from typing import List, Dict, Any
import requests
from urllib.parse import urlparse
import tempfile

# LangChain imports - updated for Hugging Face support
try:
    from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFaceHub
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
except ImportError as e:
    st.error(f"Missing dependency: {e}")
    st.stop()

# Load environment variables
from dotenv import load_dotenv
load_dotenv('.config')

# Page configuration
st.set_page_config(
    page_title="Haqdarshak Scheme Research Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

class SchemeResearchTool:
    def __init__(self):
        """Initialize the Scheme Research Tool with necessary components."""
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.documents = []
        self.urls_processed = []
        
        # Initialize OpenAI/OpenRouter components
        self.initialize_ai_components()
        
    def initialize_ai_components(self):
        """Initialize AI embeddings and LLM using OpenRouter or OpenAI."""
        try:
            # Hugging Face
            hf_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
            embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
            llm_model = os.getenv('LLM_MODEL', 'google/flan-t5-large')

            if hf_token and hf_token != 'your-huggingface-api-key':
                st.info("🤗 Using Hugging Face for embeddings and LLM")
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
                self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
                self.llm = HuggingFaceHub(
                    repo_id=llm_model,
                    huggingfacehub_api_token=hf_token,
                    model_kwargs={"temperature": 0.1, "max_length": 512}
                )
            else:
                st.error("⚠️ Please set your Hugging Face API key in the .config file!")
                st.info("💡 You can get a free Hugging Face API key from: https://huggingface.co/settings/tokens")
                st.stop()
            
        except Exception as e:
            st.error(f"Error initializing AI components: {str(e)}")
            st.stop()
    
    def load_documents_from_urls(self, urls: List[str]) -> List[Any]:
        """Load documents from URLs using appropriate loaders."""
        documents = []
        
        for url in urls:
            try:
                st.info(f"🔄 Processing URL: {url}")
                
                # Determine file type and use appropriate loader
                parsed_url = urlparse(url)
                file_extension = os.path.splitext(parsed_url.path)[1].lower()
                
                if file_extension == '.pdf':
                    # For PDFs, we need to download first
                    response = requests.get(url)
                    if response.status_code == 200:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(response.content)
                            tmp_file_path = tmp_file.name
                        
                        try:
                            loader = PyPDFLoader(tmp_file_path)
                            docs = loader.load()
                            documents.extend(docs)
                            st.success(f"✅ Successfully loaded PDF: {url}")
                        finally:
                            os.unlink(tmp_file_path)  # Clean up temp file
                    else:
                        st.error(f"❌ Failed to download PDF from {url}")
                else:
                    # For other URLs, use UnstructuredURLLoader
                    loader = UnstructuredURLLoader(urls=[url])
                    docs = loader.load()
                    documents.extend(docs)
                    st.success(f"✅ Successfully loaded URL: {url}")
                    
            except Exception as e:
                st.error(f"❌ Error processing {url}: {str(e)}")
                continue
        
        return documents
    
    def split_documents(self, documents: List[Any]) -> List[Any]:
        """Split documents into chunks for better processing."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv('CHUNK_SIZE', 1000)),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 200)),
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        st.success(f"📄 Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, chunks: List[Any]):
        """Create FAISS vector store from document chunks."""
        try:
            st.info("🔄 Creating vector embeddings...")
            
            # Create FAISS vector store
            if self.embeddings:
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
                
                # Save the vector store
                faiss_path = os.getenv('FAISS_INDEX_PATH', 'faiss_store_openai.pkl')
                with open(faiss_path, 'wb') as f:
                    pickle.dump(self.vectorstore, f)
                
                st.success(f"✅ Vector store created and saved to {faiss_path}")
                
                # Create QA chain
                self.create_qa_chain()
            else:
                st.error("❌ Embeddings not initialized")
            
        except Exception as e:
            st.error(f"❌ Error creating vector store: {str(e)}")
    
    def create_qa_chain(self):
        """Create a QA chain for answering questions."""
        try:
            if not self.vectorstore or not self.llm:
                st.error("❌ Vector store or LLM not initialized")
                return
                
            # Custom prompt template for scheme-specific questions
            prompt_template = """You are a helpful assistant for government scheme research. 
            Answer the following question based on the provided context. 
            Focus on providing accurate, relevant information about government schemes.
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            st.success("✅ QA chain created successfully!")
            
        except Exception as e:
            st.error(f"❌ Error creating QA chain: {str(e)}")
    
    def generate_summary(self) -> Dict[str, str]:
        """Generate a comprehensive summary covering the four key criteria."""
        if not self.vectorstore or not self.llm:
            return {}
        
        summary_prompts = {
            "Scheme Benefits": "What are the main benefits and advantages of this government scheme?",
            "Scheme Application Process": "What is the step-by-step application process for this scheme?",
            "Eligibility": "What are the eligibility criteria for this scheme?",
            "Documents Required": "What documents are required to apply for this scheme?"
        }
        
        summary = {}
        
        for category, prompt in summary_prompts.items():
            try:
                st.info(f"🔄 Generating summary for: {category}")
                
                # Get relevant documents
                docs = self.vectorstore.similarity_search(prompt, k=5)
                context = "\n".join([doc.page_content for doc in docs])
                
                # Generate summary using LLM
                summary_prompt = f"""Based on the following context, provide a comprehensive summary for {category}:

Context: {context}

Please provide a detailed summary focusing specifically on {category.lower()}:"""

                response = self.llm.invoke(summary_prompt)
                if hasattr(response, 'content'):
                    summary[category] = response.content.strip()
                else:
                    summary[category] = str(response).strip()
                
                st.success(f"✅ {category} summary generated")
                
            except Exception as e:
                st.error(f"❌ Error generating {category} summary: {str(e)}")
                summary[category] = "Unable to generate summary for this category."
        
        return summary
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a user question and return relevant sources."""
        if not self.qa_chain or not self.vectorstore:
            return {"error": "QA chain not initialized. Please process URLs first."}
        
        try:
            # Get answer
            result = self.qa_chain({"query": question})
            answer = result["result"]
            
            # Get relevant sources
            docs = self.vectorstore.similarity_search(question, k=3)
            sources = []
            
            for i, doc in enumerate(docs, 1):
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            return {"error": f"Error answering question: {str(e)}"}

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Haqdarshak Scheme Research Tool</h1>', unsafe_allow_html=True)
    
    # Initialize the tool
    if 'research_tool' not in st.session_state:
        st.session_state.research_tool = SchemeResearchTool()
    
    # Sidebar for URL input
    st.sidebar.markdown("## 📥 Input URLs")
    
    # URL input method
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Direct URL Input", "Upload Text File"]
    )
    
    urls = []
    
    if input_method == "Direct URL Input":
        # Direct URL input
        url_input = st.sidebar.text_area(
            "Enter URLs (one per line):",
            placeholder="https://example.com/scheme1\nhttps://example.com/scheme2",
            height=150
        )
        
        if url_input:
            urls = [url.strip() for url in url_input.split('\n') if url.strip()]
    
    else:
        # File upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload a text file with URLs (one per line):",
            type=['txt']
        )
        
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            urls = [url.strip() for url in content.split('\n') if url.strip()]
    
    # Display URLs
    if urls:
        st.sidebar.markdown("### 📋 URLs to Process:")
        for i, url in enumerate(urls, 1):
            st.sidebar.write(f"{i}. {url}")
    
    # Process URLs button
    if st.sidebar.button("🚀 Process URLs", type="primary"):
        if not urls:
            st.sidebar.error("Please enter at least one URL!")
        else:
            with st.spinner("Processing URLs..."):
                # Load documents
                documents = st.session_state.research_tool.load_documents_from_urls(urls)
                
                if documents:
                    # Split documents
                    chunks = st.session_state.research_tool.split_documents(documents)
                    
                    # Create vector store
                    st.session_state.research_tool.create_vectorstore(chunks)
                    
                    # Store processed URLs
                    st.session_state.research_tool.urls_processed = urls
                    
                    st.success("🎉 URL processing completed successfully!")
                else:
                    st.error("❌ No documents were loaded. Please check your URLs.")
    
    # Main content area
    if st.session_state.research_tool.vectorstore:
        st.markdown('<h2 class="sub-header">📊 Scheme Analysis</h2>', unsafe_allow_html=True)
        
        # Generate summary
        if st.button("📋 Generate Scheme Summary"):
            with st.spinner("Generating comprehensive summary..."):
                summary = st.session_state.research_tool.generate_summary()
                
                if summary:
                    st.markdown('<h3 class="sub-header">📋 Scheme Summary</h3>', unsafe_allow_html=True)
                    
                    for category, content in summary.items():
                        with st.expander(f"📌 {category}", expanded=True):
                            st.markdown(content)
        
        # Q&A Section
        st.markdown('<h2 class="sub-header">❓ Ask Questions</h2>', unsafe_allow_html=True)
        
        question = st.text_input(
            "Ask a question about the scheme:",
            placeholder="What are the eligibility criteria for this scheme?"
        )
        
        if st.button("🔍 Get Answer") and question:
            with st.spinner("Searching for answer..."):
                result = st.session_state.research_tool.answer_question(question)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    # Display answer
                    st.markdown("### 💡 Answer:")
                    st.markdown(result["answer"])
                    
                    # Display sources
                    if result["sources"]:
                        st.markdown("### 📚 Sources:")
                        for i, source in enumerate(result["sources"], 1):
                            with st.expander(f"Source {i}"):
                                st.markdown(f"**Content:** {source['content']}")
                                if source['metadata']:
                                    st.markdown(f"**Metadata:** {source['metadata']}")
    
    else:
        # Welcome message and instructions
        st.markdown("""
        <div class="info-box">
            <h3>🎯 Welcome to Haqdarshak Scheme Research Tool!</h3>
            <p>This tool helps you analyze government schemes by:</p>
            <ul>
                <li>📥 Loading scheme information from URLs or PDFs</li>
                <li>🔍 Creating intelligent summaries covering key criteria</li>
                <li>❓ Answering questions about scheme details</li>
                <li>📚 Providing source references for all information</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <h4>🚀 Getting Started:</h4>
            <ol>
                <li>Enter URLs in the sidebar (or upload a text file with URLs)</li>
                <li>Click "Process URLs" to load and analyze the content</li>
                <li>Generate summaries or ask specific questions</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample URL
        st.markdown("""
        <div class="success-box">
            <h4>📝 Sample URL:</h4>
            <p>You can test with this sample URL:</p>
            <code>https://mohua.gov.in/upload/uploadfiles/files/PMSVANidhi%20Guideline_English.pdf</code>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 