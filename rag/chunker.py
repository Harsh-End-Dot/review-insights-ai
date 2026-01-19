from typing import List
import tiktoken


class TextChunker:
    """
    Token-aware text chunker for RAG pipelines.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        chunk_size: int = 350,
        overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits input text into overlapping token-based chunks.
        """
        tokens = self.tokenizer.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            start += self.chunk_size - self.overlap

        return chunks
