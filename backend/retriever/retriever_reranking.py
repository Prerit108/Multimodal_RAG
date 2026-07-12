from re import search
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

class Retriever:
    def __init__(self):
        # load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")
        load_dotenv(dotenv_path = "/home/preritubuntu/RAG projext/.env")

        # Initialize ChatOpenAI pointing to OpenRouter
        self.model = ChatOpenAI(
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            max_completion_tokens=512
        )

    def start_retriever(self,combined_documents,vector_store,query:str):
        # 1. Initialize your individual retrievers
        # Example: BM25 for keyword-based retrieval
        bm25_retriever = BM25Retriever.from_documents(combined_documents)   ## combined info of all docs must be passed
        bm25_retriever.k = 5

        # Example: Vector store for semantic retrieval
        vectorstore_retriever = vector_store.as_retriever(
            search_type = "mmr",     # this enables the MMR search
            search_kwargs = {"k":5,"lambda_mult":0.7}
        )

        # 2. Combine them into an EnsembleRetriever
        # weights define the importance of each retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vectorstore_retriever],
            weights=[0.3, 0.7]
        )

        # Reranker model
        # 3. Use the reranker model
        reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base", model_kwargs={"device": "cpu"})    ## for cpu only

        # Invoke the hybrid retriever
        documents = ensemble_retriever.invoke(query)

        # Reranked scores
        pairs = [(query,doc.page_content) for doc in documents]
        scores = reranker_model.score(pairs)

        # Taking top n results
        compressor = CrossEncoderReranker(model=reranker_model, top_n=3)

        # Combine them into the ContextualCompressionRetriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever  # Your hybrid retriever from earlier
        )
        # 4. Invoke it
        # This single call retrieves, reranks, and returns the top 3 best documents!
        final_docs = compression_retriever.invoke(query)
        return final_docs

    def final_output(self,query:str,information):
        full_text = ""
        for i,doc in enumerate(information):
            full_text += f"Information {i+1} : {doc.page_content} \n" 
        prompt = f"Providing you the information about the {query} , and here is the Full information from which you have to answer\n {full_text} \n , do not add your own thinking in the answer , answer only on the basis of the provided info "
        result = self.model.invoke(prompt)
        print("*****************************Final Result od model ************************************")
        print(result.content)
        return result



