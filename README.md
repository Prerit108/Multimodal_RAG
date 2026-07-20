# Multimodal RAG Platform for Academic Papers

A high-performance, production-ready **Multimodal Retrieval-Augmented Generation (RAG)** platform designed for parsing, indexing, and reasoning over complex research papers (including text, tables, and figures/diagrams).

---

## 🌟 Key Features

* **Multimodal Document Parsing:** Uses **Docling** for structured PDF layout analysis, extracting tables, text blocks, and figure image crops.
* **Hybrid Retrieval Engine:** Combines **BM25 keyword search** and **ChromaDB vector search (MMR - Maximal Marginal Relevance)** using LangChain's `EnsembleRetriever`.
* **Cross-Encoder Semantic Reranking:** Uses `BAAI/bge-reranker-base` to score candidate passages, applying a strict **Relevance Threshold (`0.35`)** to eliminate low-scoring irrelevant chunks and prevent hallucinations.
* **Local & Cloud VLM Inference:** Supports both local VLM execution via **LM Studio** (`qwen2.5-vl-7b` / `ornith-1.0-9b`) and serverless cloud inference (`Qwen/Qwen2.5-VL-72B-Instruct`).
* **FastAPI Backend & Streamlit Frontend:** Clean REST API with streaming support and an interactive chat interface.
* **Automated Evaluation Suite (`eval.ipynb`):** Implements **RAGAS methodology** with automated synthetic test generation and LLM-as-a-Judge evaluation (Faithfulness, Answer Relevancy, Context Precision).

---

## 🏗️ System Architecture

```
                               ┌────────────────────────┐
                               │   Input PDF Papers     │
                               └───────────┬────────────┘
                                           │
                               ┌───────────▼────────────┐
                               │    Docling Parser      │ ── (Extracts Text, Tables, & Figures)
                               └───────────┬────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
         ┌──────────▼──────────┐                       ┌──────────▼──────────┐
         │     BM25 Index      │                       │ Chroma Vector Store │
         └──────────┬──────────┘                       └──────────┬──────────┘
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │
                               ┌───────────▼────────────┐
                               │   EnsembleRetriever    │
                               └───────────┬────────────┘
                                           │
                               ┌───────────▼────────────┐
                               │  BGE Cross-Encoder     │ ── (Score Filter >= 0.35)
                               └───────────┬────────────┘
                                           │
                               ┌───────────▼────────────┐
                               │  LM Studio / Cloud LLM │ ── (Synthesizes Answer)
                               └────────────────────────┘
```

---

## 🛠️ Project Structure

```
.
├── backend/
│   ├── api/
│   │   └── fast_api_backend.py     # FastAPI server exposing /chat and /upload endpoints
│   ├── embeddings/
│   │   └── embedding_generator.py # Nomic-Embed-Text v1.5 CPU embedding generator
│   ├── llms/
│   │   └── models.py              # Centralized LLM model configuration
│   ├── loaders/
│   │   └── data_loader.py         # Docling JSON loader and text chunker
│   ├── rag/
│   │   └── RAG.py                 # Full ingestion pipeline script
│   ├── retriever/
│   │   └── retriever_reranking.py # Hybrid ensemble search & CrossEncoder reranker
│   └── utils/
│       └── doc_processor.py       # Isolated document processing & VLM image analysis
├── frontend /
│   └── app.py                     # Streamlit web interface with source citations
├── chroma_db_vectorstore/         # Persistent Chroma vector store
├── eval.ipynb                     # RAGAS evaluation notebook & synthetic dataset generator
├── Paper_collector.py             # arXiv paper downloader tool
├── requirements.txt               # Dependencies list
└── README.md                      # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Clone the repository and install dependencies:
```bash
git clone https://github.com/Prerit108/Multimodal_RAG.git
cd Multimodal_RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the root directory:
```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
OPENROUTER_API_KEY=your_openrouter_key
```

Make sure **LM Studio** is running locally on port `1234` with an OpenAI-compatible endpoint.

---

### 2. Document Ingestion & Quickstart

**Option A: Quickstart with Included Sample Data (No PDF download required)**
The repository includes pre-parsed sample JSON papers in `sample_data/`. Copy them to `data/` for immediate testing:
```bash
mkdir -p data
cp sample_data/*.json data/
python backend/rag/RAG.py
```

**Option B: Full Custom Ingestion**
To download your own research papers from arXiv:
```bash
python Paper_collector.py
python backend/rag/RAG.py
```

---

### 3. Running the Backend Server

Start the FastAPI backend:
```bash
uvicorn backend.api.fast_api_backend:app --reload --port 8000
```
The API documentation will be available at `http://127.0.0.1:8000/docs`.

---

### 4. Running the Frontend Dashboard

In a new terminal, launch the Streamlit app:
```bash
streamlit run "frontend /app.py"
```
Open your browser at `http://localhost:8501`.

---

## 📊 RAG Evaluation & Benchmarking

Open [`eval.ipynb`](file:///home/preritubuntu/RAG%20projext/eval.ipynb) in Jupyter Notebook to run automated benchmarks:
1. **Synthetic Dataset Generation:** Automatically generates `(Question, Ground Truth)` pairs from randomly sampled document chunks using your local LLM.
2. **RAGAS Metric Evaluation:** Evaluates **Faithfulness**, **Answer Relevancy**, and **Context Precision** using an LLM-as-a-Judge pattern.
3. **Summary Table:** Outputs a pandas DataFrame summary of evaluation scores across all test cases.

---

## 📜 License

This project is licensed under the MIT License.
