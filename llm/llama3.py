import ollama
from typing import List


class Llama3Generator:
    """
    Llama 3 text generation wrapper for RAG pipelines.
    """

    def __init__(
        self,
        model_name: str = "llama3",
        temperature: float = 0.2,
    ):
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, query: str, context_chunks: List[str]) -> str:
        """
        Generates an answer using retrieved context.
        """
        context = "\n\n".join(context_chunks)

        prompt = f"""
You are an AI assistant that answers questions strictly using the provided context.
If the answer is not present in the context, say "I don't have enough information."

Context:
{context}

Question:
{query}

Answer:
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": self.temperature},
        )

        return response["message"]["content"].strip()
