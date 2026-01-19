from typing import List

from rag.chunker import TextChunker
from rag.embedder import OllamaEmbedder
from rag.vector_store import FaissVectorStore


class RAGRetriever:
    """
    End-to-end retriever for RAG pipelines.
    """

    def __init__(
        self,
        vector_store: FaissVectorStore,
        embedder: OllamaEmbedder,
        top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str) -> List[str]:
        """
        Retrieves the most relevant text chunks for a given query.
        """
        query_embedding = self.embedder.embed_texts([query])[0]
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k
        )
        return results
