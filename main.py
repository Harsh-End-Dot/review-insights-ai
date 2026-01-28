import os
import pandas as pd

from rag.chunker import TextChunker
from rag.embedder import OllamaEmbedder
from rag.vector_store import FaissVectorStore
from rag.retriever import RAGRetriever
from llm.llama3 import Llama3Generator


DATA_PATH = "data/reviews_clean.csv"
INDEX_PATH = "data/index"


def build_or_load_vector_store():
    """
    Builds FAISS index if not present, otherwise loads it.
    """
    embedder = OllamaEmbedder()

    # Load existing index 
    if os.path.exists(INDEX_PATH):
        print("Loading existing FAISS index...")
        store = FaissVectorStore()
        store.load(INDEX_PATH)
        return store

    #  Build index (one-time)
    print("Building FAISS index from dataset...")
    df = pd.read_csv(DATA_PATH)

    chunker = TextChunker()
    all_chunks = []

    for text in df["review_text"].dropna():
        chunks = chunker.chunk_text(text)
        all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")

    embeddings = embedder.embed_texts(all_chunks)

    store = FaissVectorStore(embedding_dim=len(embeddings[0]))
    store.add_embeddings(embeddings, all_chunks)
    store.save(INDEX_PATH)

    print("FAISS index built and saved.")
    return store


def main():
    vector_store = build_or_load_vector_store()

    embedder = OllamaEmbedder()
    retriever = RAGRetriever(vector_store, embedder)
    generator = Llama3Generator()

    print("\nReview Insights AI is ready. Type 'exit' to quit.\n")

    while True:
        query = input("User: ")
        if query.lower() == "exit":
            break

        context = retriever.retrieve(query)
        answer = generator.generate(query, context)

        print("\nAssistant:")
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    main()
