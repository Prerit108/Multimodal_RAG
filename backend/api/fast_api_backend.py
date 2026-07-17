# from notebook import full_text
# from notebook import vector_store
from fastapi import FastAPI, UploadFile, File
from pydantic import Field,BaseModel,computed_field
from typing import Literal,Annotated
from fastapi.responses import JSONResponse
import sys
import os 
import pathlib

# Dynamically add the project root directory to python path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from backend.loaders.data_loader import load_docling_json
from backend.embeddings.embedding_generator import Embedder
from backend.retriever.retriever_reranking import Retriever
from backend.utils.doc_processor import process_document
import shutil
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


app = FastAPI()


def query_only_rag(query: str) -> str:
    print("\n--- Loading All Document Chunks for Retrieval ---")
    all_docs = []
    data_dir = pathlib.Path("/home/preritubuntu/RAG projext/data")
    json_paths = sorted(list(data_dir.glob("*.json")))
    for j_path in json_paths:
        try:
            all_docs.extend(load_docling_json(j_path))
        except Exception as e:
            print(f"Error loading {j_path.name}: {e}")
            
    print(f"Loaded all {len(all_docs)} chunks from {len(json_paths)} cached papers.")

    print("\n--- Initializing QA Retriever ---")
    rag_retriever = Retriever()

    print(f"\nUser Query: '{query}'")

    print("\n--- Retrieving & Reranking Matches ---")
    # Initialize embedding and vector store using our centralized Embedder class
    embedding_model = Embedder.model_loader()
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory="/home/preritubuntu/RAG projext/chroma_db_vectorstore",
        collection_name="Multi_doc_vectorstore"
    )
    
    retrieved_docs = rag_retriever.start_retriever(all_docs, vector_store, query)
    print(f"Retrieved {len(retrieved_docs)} most relevant context passages.")

    print("\n--- Generating Final Answer from LLM ---")
    result = rag_retriever.final_output(query, retrieved_docs)

    return result.content


@app.post("/chat")
def rag_chat(query: str) -> str:
    """
    Endpoint to query over all ingested research papers currently in the database.
    """
    return query_only_rag(query)


@app.post("/query_from_doc")
def document_chat(query: str, document: UploadFile = File(...)):
    import subprocess
    dir_path =  "/home/preritubuntu/RAG projext/backend"
    file_path = os.path.join(dir_path,"user_data")
    os.makedirs(file_path,exist_ok = True)

    # Correct file locations
    document_path = os.path.join(file_path, document.filename)
    output_path = os.path.join(file_path, f"{pathlib.Path(document.filename).stem}.json")

    # Save uploaded document to disk
    with open(document_path, "wb") as buffer:
        shutil.copyfileobj(document.file, buffer)

    ## Processing document to create a structured json using a subprocess
    # This isolates memory usage and prevents CUDA/OOM interpreter crashes from shutting down VS Code.
    cmd = [
        sys.executable,
        os.path.join(dir_path, "utils", "doc_processor.py"),
        "--input", document_path,
        "--output", output_path,
        "--vlm-url", "http://127.0.0.1:1234/v1",
        "--run-vlm-on-tables"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Docling subprocess failed: {res.stderr}")

    ## Cleaning the json to get structured json
    documents = load_docling_json(json_path=output_path)

    ## Creating the embeddings 
    embedding = Embedder.model_loader()
    vector_store = Chroma(
            embedding_function=embedding,
            persist_directory = os.path.join(dir_path,"/home/preritubuntu/RAG projext/backend/user_data/Volatile_db"),
            collection_name='User_personal_doc'
        )
    chunk_id_list = []
    seen_ids = set()
    unique_documents = []
    for doc in documents:
        chunk_id = doc.metadata["chunk_id"]
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            chunk_id_list.append(chunk_id)
            unique_documents.append(doc)

    vector_store.add_documents(unique_documents, ids=chunk_id_list)

    ## Retriever
    retriever = Retriever()

    top_results = retriever.start_retriever(documents,vector_store,query,top_n_values = 4)

    final_answer = retriever.final_output(query,top_results)

    return final_answer 
    