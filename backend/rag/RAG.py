import sys
from pathlib import Path
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import subprocess

# Hotfix: Map langchain_classic.retrievers to langchain.retrievers to resolve import conflicts in langchain_community
try:
    import langchain_classic.retrievers
except ImportError:
    try:
        import langchain_classic.retrievers
        import langchain_classic.retrievers.document_compressors
        import langchain_classic.retrievers.document_compressors.cross_encoder
        sys.modules['langchain.retrievers'] = langchain_classic.retrievers
        sys.modules['langchain.retrievers.document_compressors'] = langchain_classic.retrievers.document_compressors
        sys.modules['langchain.retrievers.document_compressors.cross_encoder'] = langchain_classic.retrievers.document_compressors.cross_encoder
    except ImportError:
        pass

# Dynamically add the project root directory to python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.loaders.data_loader import load_docling_json
from backend.embeddings.embedding_generator import Embedder
from backend.retriever.retriever_reranking import Retriever

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    papers_dir = project_root / "papers"
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    # print("--- Step 1: Scanning for Research Papers ---")
    # if not papers_dir.exists():
    #     print(f"Error: Papers directory not found at {papers_dir}")
    #     sys.exit(1)

    # pdf_paths = sorted(list(papers_dir.glob("*.pdf")))
    # if not pdf_paths:
    #     print("No PDF files found in the papers/ directory.")
    #     sys.exit(1)
        
    # print(f"Found {len(pdf_paths)} PDFs in {papers_dir.name}/.")

    # print("\n--- Step 2: Parsing Documents (Phase 1) ---")
    
    # # Configuration for VLM scanning during ingestion
    # vlm_url = "http://127.0.0.1:1234/v1"  # Local VLM URL
    # run_vlm_on_tables = True              # Enable table cropping/scanning
    
    # heavy_parse_count = 0
    # for idx, pdf_path in enumerate(pdf_paths, 1):
    #     json_path = data_dir / f"{pdf_path.stem}.json"
        
    #     # Parse using Docling in a subprocess if not already parsed
    #     # This isolates GPU/RAM memory usage and fully releases it on completion
    #     if not json_path.exists():
    #         print(f"\n[{idx}/{len(pdf_paths)}] Parsing: {pdf_path.name}")
    #         try:
    #             print(f"  Parsing with Docling (GPU accelerated subprocess)...")
    #             cmd = [
    #                 sys.executable,
    #                 str(project_root / "backend" / "utils" / "doc_processor.py"),
    #                 "--input", str(pdf_path),
    #                 "--output", str(json_path)
    #             ]
    #             if vlm_url:
    #                 cmd.extend(["--vlm-url", vlm_url])
    #             if run_vlm_on_tables:
    #                 cmd.append("--run-vlm-on-tables")
                
    #             # Execute parsing subprocess
    #             res = subprocess.run(cmd, capture_output=True, text=True)
    #             if res.returncode != 0:
    #                 print(f"  Error parsing {pdf_path.name}: {res.stderr}")
    #             else:
    #                 print(f"  Successfully parsed {pdf_path.name}!")
                    
    #             # Sleep for 10 seconds after every 2 active PDF parses to prevent overheating
    #             heavy_parse_count += 1
    #             if heavy_parse_count % 2 == 0:
    #                 print("  [Cooldown] Sleeping for 10 seconds to prevent GPU/CPU overheating...")
    #                 import time
    #                 time.sleep(10)
    #         except Exception as e:
    #             print(f"  Subprocess error for {pdf_path.name}: {e}")
    #             continue
    #     else:
    #         # Output minimal print for cached documents so output isn't cluttered
    #         pass

    # print("\n--- Step 3: Initializing Embeddings & Vector Store (Phase 2) ---")
    # # Load the embedding model (nomic-ai/nomic-embed-text-v1.5) now that parsing is done
    # # This prevents loading the 1.5GB embedding model into RAM while Docling is active!
    embedding_model = Embedder.model_loader()
    
    # # # Initialize/connect to Chroma DB
    vector_store = Embedder.vector_db_creator(embedding_model)

    # print("\n--- Step 4: Indexing Parsed Documents ---")
    # heavy_index_count = 0
    # for idx, pdf_path in enumerate(pdf_paths, 1):
    #     json_path = data_dir / f"{pdf_path.stem}.json"
    #     doc_id = pdf_path.stem
        
    #     if not json_path.exists():
    #         continue
            
    #     # Check if already indexed in vector store
    #     try:
    #         existing = vector_store.get(where={"doc_id": doc_id})
    #         if existing and existing.get("ids"):
    #             # Already indexed, skip
    #             continue
            
    #         print(f"\n[{idx}/{len(pdf_paths)}] Indexing: {pdf_path.name}")
    #         # Load partition into LangChain Document chunks only if we need to index it
    #         paper_docs = load_docling_json(json_path)
    #         print(f"  Indexing {len(paper_docs)} chunks in Chroma DB...")
    #         Embedder.generate_embedding(paper_docs, vector_store)
    #         print(f"  Successfully indexed chunks!")
            
    #         # Sleep for 10 seconds after every 2 active PDF indexing steps to prevent overheating
    #         heavy_index_count += 1
    #         if heavy_index_count % 2 == 0:
    #             print("  [Cooldown] Sleeping for 10 seconds to prevent GPU/CPU overheating...")
    #             import time
    #             time.sleep(10)
    #     except Exception as e:
    #         print(f"  Error indexing in Chroma: {e}")

    # Step 4: Load all parsed JSONs at the very end to construct unified BM25 index
    print("\n--- Step 4: Loading All Document Chunks for Retrieval ---")
    all_docs = []
    json_paths = sorted(list(data_dir.glob("*.json")))
    for j_path in json_paths:
        try:
            all_docs.extend(load_docling_json(j_path))
        except Exception as e:
            print(f"Error loading {j_path.name}: {e}")
            
    print(f"Loaded all {len(all_docs)} chunks from {len(json_paths)} cached papers.")

    print("\n--- Step 5: Initializing QA Retriever ---")
    # Initialize the QA retriever and generation model
    rag_retriever = Retriever()

    # 6. Define query to test the RAG system
    query = "what is masked attention ? "
    print(f"\nUser Query: '{query}'")

    print("\n--- Step 6: Retrieving & Reranking Matches ---")
    # Run Hybrid retrieval (BM25 + Semantic MMR) followed by BGE Reranker
    retrieved_docs = rag_retriever.start_retriever(all_docs, vector_store, query)
    print(f"Retrieved {len(retrieved_docs)} most relevant context passages.")

    print("\n--- Step 7: Generating Final Answer from LLM ---")
    # Pass the context passages to Qwen/Gemma to generate the final synthesized response
    rag_retriever.final_output(query, retrieved_docs)









