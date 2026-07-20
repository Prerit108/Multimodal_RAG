import subprocess
import streamlit as st
import requests
import json
import os

# Set page configuration with standard title
st.set_page_config(
    page_title="Multimodal RAG System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base URL for FastAPI backend connection
API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")

# Custom CSS styling for modern UI without emojis
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    
    .chat-user {
        background-color: #2563eb;
        color: #ffffff;
        padding: 0.9rem 1.2rem;
        border-radius: 12px 12px 0px 12px;
        margin: 0.6rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .chat-assistant {
        background-color: #1e293b;
        color: #f8fafc;
        padding: 0.9rem 1.2rem;
        border-radius: 12px 12px 12px 0px;
        margin: 0.6rem 0;
        max-width: 85%;
        margin-right: auto;
        border: 1px solid #334155;
    }

    .sources-box {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 0.8rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

# Application Title & Subtitle
st.markdown('<div class="main-header">Multimodal RAG Research System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Scalable paper retrieval, dynamic document analysis, and conversational memory.</div>', unsafe_allow_html=True)

# Initialize Chat History Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar Configuration Panel
with st.sidebar:
    st.header("Control Panel")
    
    # 1. Search Scope Configuration
    search_scope = st.radio(
        "Search Target Scope:",
        ["Database Only", "Uploaded Docs Only", "Combined Search"],
        index=0,
        help="Select which document collection to retrieve context from."
    )
    
    # Map search scope selection to backend API key
    search_mode_map = {
        "Database Only": "db_only",
        "Uploaded Docs Only": "uploaded_only",
        "Combined Search": "combined"
    }
    selected_search_mode = search_mode_map[search_scope]
    
    st.divider()
    
    # 2. Parsing Speed Configuration
    fast_mode_toggle = st.checkbox(
        "Enable Fast Parsing Mode",
        value=True,
        help="When enabled, PDF parsing takes 2-5 seconds (text only). Disable for VLM table analysis."
    )
    
    st.divider()
    
    # 3. File Upload — max_uploads=5 enforces the limit at the widget level
    st.subheader("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload up to 5 PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload user documents for dynamic analysis. Maximum 5 PDFs."
    )
    if uploaded_files and len(uploaded_files) > 5:
        uploaded_files = uploaded_files[:5]

# Display Past Chat History
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f'<div class="chat-user"><b>User:</b><br>{chat["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-assistant"><b>Assistant:</b><br>{chat["content"]}</div>', unsafe_allow_html=True)
        
        # Display Sources if present in message data — [Source N] matches LLM citation numbers
        if "sources" in chat and chat["sources"]:
            with st.expander("References and Sources"):
                for src in chat["sources"]:
                    src_num = src.get("source_number", "?")
                    pg_str = f" — Page {src['page']}" if src.get("page") else ""
                    st.write(f"[Source {src_num}] {src['filename']}{pg_str}")

# Query Input Area
user_input = st.chat_input("Enter your question here...")

if user_input:
    # Render user prompt in UI
    st.markdown(f'<div class="chat-user"><b>User:</b><br>{user_input}</div>', unsafe_allow_html=True)
    
    # Build conversation history payload for API
    formatted_history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.chat_history
    ]
    
    # Save user message to session state
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("Processing request and retrieving context..."):
        try:
            # Handle API routing based on mode selection
            if selected_search_mode == "db_only" and not uploaded_files:
                # Direct JSON payload for database-only search
                payload = {
                    "query": user_input,
                    "history": formatted_history
                }
                response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=180)
            else:
                # Multipart form payload for upload/combined search modes
                files_payload = []
                if uploaded_files:
                    for f in uploaded_files:
                        files_payload.append(
                            ("documents", (f.name, f.getvalue(), "application/pdf"))
                        )
                
                form_data = {
                    "query": user_input,
                    "fast_mode": str(fast_mode_toggle),
                    "search_mode": selected_search_mode,
                    "history_json": json.dumps(formatted_history)
                }
                
                response = requests.post(
                    f"{API_BASE_URL}/query_from_doc",
                    data=form_data,
                    files=files_payload if files_payload else None,
                    timeout=300
                )
                
            if response.status_code == 200:
                res_data = response.json()
                answer_text = res_data.get("answer", "")
                source_list = res_data.get("sources", [])
                
                # Render Assistant Response
                st.markdown(f'<div class="chat-assistant"><b>Assistant:</b><br>{answer_text}</div>', unsafe_allow_html=True)
                
                # Render Sources — [Source N] label matches the citation number in the LLM answer
                if source_list:
                    with st.expander("References and Sources"):
                        for src in source_list:
                            src_num = src.get("source_number", "?")
                            pg_str = f" — Page {src['page']}" if src.get("page") else ""
                            st.write(f"[Source {src_num}] {src['filename']}{pg_str}")
                            
                # Save assistant response to session state history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": source_list
                })
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
                
        except Exception as e:
            st.error(f"Failed to connect to the backend API: {str(e)}")
