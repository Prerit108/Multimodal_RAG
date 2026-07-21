import sys
import os
import json
import shutil
import subprocess
import pathlib
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

# Dynamically resolve and append the project root directory to Python path
project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from backend.loaders.data_loader import load_docling_json
from backend.embeddings.embedding_generator import Embedder
from backend.retriever.retriever_reranking import Retriever
from langchain_chroma import Chroma

app = FastAPI()

# Canonical prefix of the refusal message produced by final_output when information is empty.
# Used to detect when the LLM had no useful context and suppress misleading source citations.
REFUSAL_PREFIX = "No relevant information was found"


class ChatRequest(BaseModel):
    """Request model for standard database chat with history."""
    query: str
    history: Optional[List[dict]] = []


def format_sources(retrieved_docs) -> List[dict]:
    """
    Extracts source references preserving the exact [Source N] numbers the LLM
    sees in its system prompt. The LLM numbers chunks as [Source 1], [Source 2], etc.
    matching the enumerate(information) index in final_output(), so source_number
    here is guaranteed to match what the LLM cited in its answer.
    """
    sources = []
    for i, doc in enumerate(retrieved_docs):
        fn = doc.metadata.get("filename") or doc.metadata.get("doc_id")
        pg = doc.metadata.get("page") or doc.metadata.get("page_start")
        if fn:
            sources.append({
                "source_number": i + 1,   # matches [Source N] in LLM answer
                "filename": str(fn),
                "page": pg
            })
    return sources


@app.post("/chat")
def rag_chat(request: ChatRequest):
    """
    Endpoint to query across all ingested research papers in the main database.
    Accepts user query and optional chat history.
    """
    query = request.query
    history = request.history

    # Step 1: Load all parsed JSON chunks needed for BM25 keyword retrieval.
    # Note: Chroma handles semantic search from disk; BM25 needs docs in RAM.
    all_docs = []
    data_dir = project_root / "data"
    json_paths = sorted(list(data_dir.glob("*.json")))
    for j_path in json_paths:
        try:
            all_docs.extend(load_docling_json(j_path))
        except Exception as e:
            print(f"Error loading {j_path.name}: {e}")

    if not all_docs:
        return {"answer": "No documents found in the knowledge base. Please run RAG.py to index papers first.", "sources": []}

    # Step 2: Connect to the primary persistent Chroma vector store (no reload, reads from disk)
    embedding_model = Embedder.model_loader()
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=str(project_root / "chroma_db_vectorstore"),
        collection_name="Multi_doc_vectorstore"
    )

    # Step 3: Retrieve relevant document passages and generate LLM answer
    rag_retriever = Retriever()
    retrieved_docs = rag_retriever.start_retriever(all_docs, vector_store, query)
    result = rag_retriever.final_output(query, retrieved_docs, history=history)

    # Step 4: Only return source citations when the LLM gave a real answer.
    # If it returned the refusal message, sources would be misleading (retriever always
    # returns top-K docs even when they are completely irrelevant to the query).
    answer_text = result.content
    sources = [] if answer_text.startswith(REFUSAL_PREFIX) else format_sources(retrieved_docs)

    return {
        "answer": answer_text,
        "sources": sources
    }


@app.post("/query_from_doc")
def document_chat(
    query: str = Form(...),
    fast_mode: bool = Form(True),
    search_mode: str = Form("uploaded_only"),
    history_json: Optional[str] = Form(None),
    documents: List[UploadFile] = File(None)
):
    """
    Endpoint for uploading up to 5 documents, parsing them on-the-fly, and searching
    via Database Only, Uploaded Docs Only, or Combined search modes.
    """
    # Parse chat history JSON string if provided
    history = []
    if history_json:
        try:
            history = json.loads(history_json)
        except Exception as e:
            print(f"Error parsing history_json: {e}")

    dir_path = project_root / "backend"
    user_data_dir = dir_path / "user_data"
    user_data_dir.mkdir(parents=True, exist_ok=True)

    parsed_docs = []

    # Step 1: Process uploaded PDF documents if present and relevant to the search mode
    # Enforce server-side maximum of 5 documents regardless of what the client sends
    if documents:
        documents = documents[:5]
    if documents and search_mode in ["uploaded_only", "combined"]:
        for doc in documents:
            if not doc.filename:
                continue

            doc_path = user_data_dir / doc.filename
            out_json_path = user_data_dir / f"{doc_path.stem}.json"

            # Save uploaded document stream to temporary storage
            with doc_path.open("wb") as buffer:
                shutil.copyfileobj(doc.file, buffer)

            # Check if JSON parse already exists to avoid re-parsing on every query
            if out_json_path.exists():
                chunks = load_docling_json(json_path=out_json_path)
            else:
                # Construct subprocess command for memory-isolated parsing
                cmd = [
                    sys.executable,
                    str(dir_path / "utils" / "doc_processor.py"),
                    "--input", str(doc_path),
                    "--output", str(out_json_path),
                    "--vlm-url", "http://127.0.0.1:1234/v1"
                ]

                # In Deep Scan mode (fast_mode off): enable VLM on both tables and images.
                # In Fast Mode (fast_mode on): skip all VLM scanning entirely.
                # fast_mode arrives as a string from HTTP form — handle all string variants.
                fast_mode_is_off = str(fast_mode).lower() in ("false", "0", "no")
                if fast_mode_is_off:
                    cmd.append("--run-vlm-on-tables")
                    cmd.append("--run-vlm-on-images")

                # Execute Docling parser subprocess
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    print(f"Error parsing {doc.filename}: {res.stderr}")
                    continue

                # Load extracted JSON chunks
                chunks = load_docling_json(json_path=out_json_path)

            parsed_docs.extend(chunks)

    # Initialize CPU embedding model
    embedding_model = Embedder.model_loader()

    # Step 2: Prepare main database vector store and document chunks (Only if querying Main DB)
    main_docs = []
    main_vector_store = None
    if search_mode in ["db_only", "combined"]:
        data_dir = project_root / "data"
        json_paths = sorted(list(data_dir.glob("*.json")))
        for j_path in json_paths:
            try:
                main_docs.extend(load_docling_json(j_path))
            except Exception as e:
                pass

        main_vector_store = Chroma(
            embedding_function=embedding_model,
            persist_directory=str(project_root / "chroma_db_vectorstore"),
            collection_name="Multi_doc_vectorstore"
        )

    # Step 3: Index uploaded document chunks into IN-MEMORY Chroma vector store (Only if querying Uploaded Docs)
    uploaded_vector_store = None
    if parsed_docs and search_mode in ["uploaded_only", "combined"]:
        # Deduplicate chunk IDs to avoid database insert errors
        seen_ids = set()
        unique_docs = []
        for d in parsed_docs:
            cid = d.metadata.get("chunk_id") or d.page_content[:50]
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_docs.append(d)

        if unique_docs:
            uploaded_vector_store = Chroma.from_documents(
                documents=unique_docs,
                embedding=embedding_model
            )

    # Step 4: Execute retrieval based on selected search_mode
    rag_retriever = Retriever()
    retrieved_docs = []

    if search_mode == "db_only":
        retrieved_docs = rag_retriever.start_retriever(main_docs, main_vector_store, query)
    elif search_mode == "uploaded_only":
        if uploaded_vector_store and parsed_docs:
            retrieved_docs = rag_retriever.start_retriever(parsed_docs, uploaded_vector_store, query,top_n_values = 5)
        else:
            retrieved_docs = []
    else:  # Combined search
        docs_db = rag_retriever.start_retriever(main_docs, main_vector_store, query)
        docs_up = []
        if uploaded_vector_store and parsed_docs:
            docs_up = rag_retriever.start_retriever(parsed_docs, uploaded_vector_store, query,top_n_values = 5)

        # Merge results from both pools without duplicates
        seen_contents = set()
        for doc in docs_up + docs_db:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                retrieved_docs.append(doc)

    # Step 5: Synthesize LLM answer with chat history context
    result = rag_retriever.final_output(query, retrieved_docs, history=history)

    # Only return source citations when the LLM actually gave a real grounded answer.
    # Suppress sources if it returned the refusal message (no relevant context found).
    answer_text = result.content
    sources = [] if answer_text.startswith(REFUSAL_PREFIX) else format_sources(retrieved_docs)

    return {
        "answer": answer_text,
        "sources": sources
    }

