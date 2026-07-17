import streamlit as st
import requests
import os

# Set page config with a custom title and icon
st.set_page_config(
    page_title="Multimodal RAG Research Assistant",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL configuration
API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")

# Premium custom CSS styling
st.markdown("""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title styling */
    .title-container {
        padding: 1.5rem 0rem;
        text-align: center;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Chat layout styling */
    .user-bubble {
        background-color: #3b82f6;
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 15px 15px 0px 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.2);
    }
    
    .assistant-bubble {
        background-color: #1e293b;
        color: #f1f5f9;
        padding: 0.8rem 1.2rem;
        border-radius: 15px 15px 15px 0px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-right: auto;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-weight: 600;
        font-size: 1.2rem;
        color: #f8fafc;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Display
st.markdown('<div class="title-container">Multimodal RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Query over your research paper library or analyze a single document on-the-fly.</div>', unsafe_allow_html=True)

# Sidebar Navigation / Mode Selection
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Choose Mode:",
        ["📚 Multi-Doc Library Chat", "📄 Single-Doc Deep Dive"],
        index=0
    )
    
    st.divider()
    st.markdown("""
    **Pipeline Status**
    *   ⚡ **Embedding Device:** CPU
    *   🧠 **LLM / VLM:** LM Studio (`gemma-3-4b`)
    *   📦 **Vector DB:** Chroma
    """)
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# MODE 1: Multi-Doc Library Chat
if mode == "📚 Multi-Doc Library Chat":
    st.subheader("📚 Query Research Paper Database")
    st.info("This query retrieves information across all research papers currently ingested in the Chroma database.")
    
    # Display Chat History
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f'<div class="user-bubble">🧑‍💻 <b>You:</b><br>{chat["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-bubble">🤖 <b>Assistant:</b><br>{chat["content"]}</div>', unsafe_allow_html=True)
            
    # Input Area
    user_query = st.chat_input("Ask a question about the papers...")
    
    if user_query:
        # Display user query
        st.markdown(f'<div class="user-bubble">🧑‍💻 <b>You:</b><br>{user_query}</div>', unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.spinner("Analyzing papers and generating response..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    params={"query": user_query},
                    timeout=180
                )
                if response.status_code == 200:
                    answer = response.text
                    # Strip wrapping quotes if return string has them
                    if answer.startswith('"') and answer.endswith('"'):
                        answer = answer[1:-1].replace('\\n', '\n')
                    
                    st.markdown(f'<div class="assistant-bubble">🤖 <b>Assistant:</b><br>{answer}</div>', unsafe_allow_html=True)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to the backend API: {e}")

# MODE 2: Single-Doc Deep Dive
else:
    st.subheader("📄 Dynamic Document Analysis")
    st.info("Upload a PDF to parse it, index it in a temporary local store, and run queries strictly on this file.")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    
    if uploaded_file:
        st.success(f"✓ '{uploaded_file.name}' ready for processing.")
        
        # Input Area for single-doc queries
        doc_query = st.text_input("Ask a question about this document:")
        
        if st.button("Submit Query", type="primary") and doc_query:
            with st.spinner("Processing document (parsing tables/figures & generating search index)..."):
                try:
                    # Prepare file payload
                    files = {
                        "document": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
                    }
                    
                    response = requests.post(
                        f"{API_BASE_URL}/query_from_doc",
                        params={"query": doc_query},
                        files=files,
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        try:
                            res_json = response.json()
                        except Exception:
                            res_json = response.text

                        st.write("---")
                        st.markdown(f"### 🤖 Response for: *{uploaded_file.name}*")
                        
                        if isinstance(res_json, dict):
                            if "error" in res_json:
                                st.error(res_json["error"])
                            else:
                                # Retrieve answer from response or content key
                                answer = res_json.get("response", res_json.get("content", str(res_json)))
                                st.markdown(answer)
                                
                                if "chunks_count" in res_json:
                                    with st.expander("🔍 Show Metadata Details"):
                                        st.write(f"**Extracted Text Chunks:** {res_json['chunks_count']}")
                        else:
                            answer = str(res_json)
                            if answer.startswith('"') and answer.endswith('"'):
                                answer = answer[1:-1].replace('\\n', '\n')
                            st.markdown(answer)
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"An error occurred while connecting to the backend: {str(e)}")
