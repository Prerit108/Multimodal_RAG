from re import search
import math
try:
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pathlib import Path
from backend.llms.models import LM_STUDIO_MODEL


class Retriever:
    # Class-level variables to cache objects across API requests in FastAPI
    _reranker_model = None
    _db_bm25_retriever = None
    _cached_docs_len = 0
    _cached_first_content = ""
    _cached_last_content = ""

    def __init__(self):
        load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

        # Initialize ChatOpenAI pointing to OpenRouter
        # self.model = ChatOpenAI(
        #     model="nvidia/nemotron-3-ultra-550b-a55b:free",
        #     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        #     base_url="https://openrouter.ai/api/v1",
        #     max_completion_tokens=2000
        # )
        self.model = ChatOpenAI(
            base_url= "http://127.0.0.1:1234/v1",
            openai_api_key="lm-studio",  # A placeholder key is required by LangChain
            model= LM_STUDIO_MODEL,
        )

        # Preload the CrossEncoder reranker model once as a singleton class attribute
        if Retriever._reranker_model is None:
            Retriever._reranker_model = HuggingFaceCrossEncoder(
                model_name="BAAI/bge-reranker-base",
                model_kwargs={"device": "cpu"}
            )

    def start_retriever(self,combined_documents,vector_store,query:str,top_n_values:int = 15):
        # 1. Check if we can reuse the cached BM25 retriever to avoid tokenizing thousands of docs on every query
        is_same = False
        if (Retriever._db_bm25_retriever is not None and 
            len(combined_documents) == Retriever._cached_docs_len and 
            len(combined_documents) > 0):
            if (combined_documents[0].page_content == Retriever._cached_first_content and 
                combined_documents[-1].page_content == Retriever._cached_last_content):
                is_same = True

        if is_same:
            bm25_retriever = Retriever._db_bm25_retriever
        else:
            bm25_retriever = BM25Retriever.from_documents(combined_documents)
            bm25_retriever.k = top_n_values
            # Cache the retriever index details
            Retriever._db_bm25_retriever = bm25_retriever
            Retriever._cached_docs_len = len(combined_documents)
            Retriever._cached_first_content = combined_documents[0].page_content if combined_documents else ""
            Retriever._cached_last_content = combined_documents[-1].page_content if combined_documents else ""

        # 2. Vector store for semantic retrieval
        vectorstore_retriever = vector_store.as_retriever(
            search_type = "mmr",     # this enables the MMR search
            search_kwargs = {"k":top_n_values,"lambda_mult":0.7}
        )

        # 3. Combine them into an EnsembleRetriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vectorstore_retriever],
            weights=[0.3, 0.7]
        )

        # 4. Rerank the top results using the preloaded class-level model
        top_n = math.ceil(top_n_values)
        documents = ensemble_retriever.invoke(query)

        if not documents:
            return []

        # Score candidates with the CrossEncoder model
        pairs = [(query, doc.page_content) for doc in documents]
        scores = Retriever._reranker_model.score(pairs)

        # Pair documents with their reranker scores
        doc_score_pairs = list(zip(documents, scores))

        # Filter out irrelevant chunks whose score falls below the relevance threshold (0.35)
        RELEVANCE_THRESHOLD = 0.35
        filtered_pairs = [dp for dp in doc_score_pairs if dp[1] >= RELEVANCE_THRESHOLD]

        # Sort filtered documents in descending order of score
        filtered_pairs.sort(key=lambda x: x[1], reverse=True)

        # Select top N most relevant documents
        final_docs = [dp[0] for dp in filtered_pairs[:top_n]]

        return final_docs

    def final_output(self, query: str, information, history: list = None):
        """
        Generates the final synthesized answer from the LLM based on retrieved
        documents, formatting the prompt using structured LangChain message objects.
        """
        # Guard: if no documents were retrieved or all failed score threshold, return refusal immediately.
        if not information:
            no_info_msg = (
                "No relevant information was found in the knowledge base to answer this query. "
                "Please try rephrasing your question or check that the relevant documents have been indexed."
            )
            print("*****************************Final Result of model ************************************")
            print(no_info_msg)
            return AIMessage(content=no_info_msg)

        # Format the retrieved context segments with explicit boundaries
        full_text = ""
        for i, doc in enumerate(information):
            full_text += f"[Source {i+1}]: {doc.page_content}\n\n"

        # Construct structured message queue for the LLM
        messages = []

        # Define a strict system instruction with strict hallucination guards
        system_instruction = f"""You are a research assistant that answers questions STRICTLY and EXCLUSIVELY from the provided source excerpts below.

            STRICT RULES — you MUST follow these without exception:
            1. Answer ONLY using facts directly stated in the source excerpts below.
            2. If the excerpts do NOT contain the answer to the user's question, reply EXACTLY with: "The provided documents do not contain information to answer this question."
            3. Do NOT use your pre-trained internal knowledge, general background knowledge, or external facts under ANY circumstances.
            4. Do NOT speculate, infer, or extrapolate concepts (e.g. filters, kernels, mechanisms) that are not explicitly defined in the source excerpts.
            5. Cite the source number (e.g. [Source 1]) when referencing specific facts.

            ---
            SOURCE EXCERPTS:
            {full_text}---"""
            
        messages.append(SystemMessage(content=system_instruction))

        # Add formatted history messages if history context is available
        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))

        # Add the active user query as the final human message
        messages.append(HumanMessage(content=query))

        # Invoke ChatOpenAI with message sequence

        # for chunk in self.model.stream(messages):
        #     print(chunk.content, end="", flush=True)
        result = self.model.invoke(messages)
        print("*****************************Final Result of model ************************************")
        print(result)
        return result



