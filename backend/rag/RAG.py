import sys
from pathlib import Path
# Dynamically add the project root directory to python path
sys.path.append(str(Path(__file__).resolve().parents[2]))


from backend.utils.doc_processor import process_document
from backend.loaders.data_loader import load_docling_json
from backend.embeddings.embedding_generator import Embedder
from backend.retriever.retriever_reranking import Retriever

if __name__ == "__main__":
    # 1. Path to the parsed document JSON file
    # We use 'parsed_sections_docling.json' located in the project root folder
    project_root = Path(__file__).resolve().parents[2]
    json_path = project_root / "parsed_sections_docling.json"
    
    print("--- Step 1: Loading and Partitioning Document JSON ---")
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        print("Please run the document processor first to generate this file.")
        sys.exit(1)
        
    docs = load_docling_json(json_path)
    
    print("\n--- Step 2: Initializing Embeddings & Vector Store ---")
    # Load the embedding model (OpenAI's text-embedding-3-small via OpenRouter)
    embedding_model = Embedder.model_loader()
    
    # Initialize/connect to Chroma DB
    vector_store = Embedder.vector_db_creator(embedding_model)
    
    print("\n--- Step 3: Indexing Documents in Vector Store ---")
    # Generate embeddings and add to vector store
    Embedder.generate_embedding(docs, vector_store)
    print("Documents successfully indexed!")
    
    print("\n--- Step 4: Initializing QA Retriever ---")
    # Initialize the QA retriever and generation model
    rag_retriever = Retriever()
    
    # 5. Define query to test the RAG system
    query = "What is the role of self-attention in transformers?"
    print(f"\nUser Query: '{query}'")
    
    print("\n--- Step 5: Retrieving & Reranking Matches ---")
    # Run Hybrid retrieval (BM25 + Semantic MMR) followed by BGE Reranker
    retrieved_docs = rag_retriever.start_retriever(docs, vector_store, query)
    print(f"Retrieved {len(retrieved_docs)} most relevant context passages.")
    
    print("\n--- Step 6: Generating Final Answer from LLM ---")
    # Pass the context passages to Qwen/Gemma to generate the final synthesized response
    rag_retriever.final_output(query, retrieved_docs)








