from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class Embedder:

    def model_loader():
        # Initialize a free local embedding model running on CPU
        embedding = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            # model_kwargs={"device": "cpu"}
        )
        return embedding


    # 3. Initialize Chroma
    def vector_db_creator(embedding):
        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory='chroma_db_vectorstore',
            collection_name='Multi_doc_vectorstore'
        )

        return vector_store

    def generate_embedding(documents,vector_store):
        chunk_id_list = []
        for doc in documents:
            # print(doc.metadata["chunk_id"])
            chunk_id_list.append(doc.metadata["chunk_id"])

        vector_store.add_documents(documents,ids = chunk_id_list)



