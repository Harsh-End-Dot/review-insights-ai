from typing import List
import faiss
import numpy as np
import pickle
import os


class FaissVectorStore:
    """
    FAISS-based vector store for similarity search.
    """

    def __init__(self, embedding_dim: int):
        """
        Initialize the FAISS index.

        Args:
            embedding_dim (int): Dimension of the embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.text_chunks: List[str] = []

    def add_embeddings(self, embeddings: List[List[float]], texts: List[str]):
        """
        Add embeddings and their corresponding text chunks to the FAISS index.
        """
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.text_chunks.extend(texts)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """
        Search for the most similar text chunks to a query embedding.
        """
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        return [self.text_chunks[i] for i in indices[0]]

    def save(self, path: str):
        """
        Save FAISS index and metadata to disk.
        """
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "texts.pkl"), "wb") as f:
            pickle.dump(self.text_chunks, f)

    def load(self, path: str):
        """
        Load FAISS index and metadata from disk.
        """
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "texts.pkl"), "rb") as f:
            self.text_chunks = pickle.load(f)
