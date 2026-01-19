from typing import List
import ollama


class OllamaEmbedder:
    """
    Embedding generator using Ollama models.
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of text chunks using Ollama.
        """
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            for text in batch:
                response = ollama.embeddings(
                    model=self.model_name,
                    prompt=text
                )
                embeddings.append(response["embedding"])

        return embeddings
