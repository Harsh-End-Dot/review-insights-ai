import streamlit as st

from rag.embedder import OllamaEmbedder
from rag.retriever import RAGRetriever
from rag.vector_store import FaissVectorStore
from llm.llama3 import Llama3Generator
from observability.monitor import (
    track_time,
    log_retrieval_metrics,
    log_generation_start,
    log_generation_end,
)


INDEX_PATH = "data/index"

st.set_page_config(page_title="Review Insights AI", layout="wide")
st.title(" Review Insights AI")

@st.cache_resource
def load_components():
    store = FaissVectorStore()
    store.load(INDEX_PATH)

    embedder = OllamaEmbedder()
    retriever = RAGRetriever(store, embedder)
    generator = Llama3Generator()

    return retriever, generator


retriever, generator = load_components()

query = st.text_input("Ask a question about product reviews:")

if query:
    with st.spinner("Thinking..."):
        with track_time("Retrieval"):
            context = retriever.retrieve(query)
            log_retrieval_metrics(query, len(context))

        with track_time("Generation"):
            log_generation_start()
            answer = generator.generate(query, context)
            log_generation_end()

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved Context"):
        for c in context:
            st.write("- ", c)
