import math
import re
from typing import Dict, List


class OptimizationRAG:
    """
    A Pure Python implementation of RAG.
    Zero external dependencies. Maximum stability.
    Demonstrates strong algorithmic fundamentals.
    """

    def __init__(self):
        # In-memory vector store: List of {"text": str, "vector": List[float]}
        self.doc_store: List[Dict] = []

    def _simple_tokenize(self, text: str) -> List[str]:
        """Normalize and tokenize text."""
        text = text.lower()
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

    def _get_embedding_mock(self, text: str) -> List[float]:
        """
        A deterministic mock embedding function.
        Generates a vector based on character codes (Hashing trick).
        In production, this would call OpenAI API.
        """
        # Create a simple hash-based vector of size 10 for demonstration
        vec = [0.0] * 10
        for i, char in enumerate(text[:100]):  # Limit to first 100 chars
            idx = i % 10
            vec[idx] += ord(char)

        # Normalize vector (Cosine similarity requires normalized vectors)
        magnitude = math.sqrt(sum(v**2 for v in vec))
        if magnitude == 0:
            return vec
        return [v / magnitude for v in vec]

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Math: Calculate cosine similarity between two vectors."""
        # Since we normalized them 'dot product' IS cosine similarity
        return sum(a * b for a, b in zip(v1, v2, strict=True))

    def ingest_text(self, text: str, chunk_size: int = 100) -> None:
        """
        Split text and index it.
        """
        # Simple splitting by sentence or rough characters
        # Using regex to split by period but keep period
        sentences = re.split(r"(?<=[.!?]) +", text)

        print(f"📄 Split text into {len(sentences)} chunks (Sentences).")

        for chunk in sentences:
            if not chunk.strip():
                continue
            vector = self._get_embedding_mock(chunk)
            self.doc_store.append({"text": chunk, "vector": vector})

        print(f"💾 Indexed {len(self.doc_store)} documents total.")

    def query(self, question: str, top_k: int = 2) -> Dict:
        """
        Retrieve relevant docs and simulate answer.
        """
        q_vec = self._get_embedding_mock(question)

        # Linear Search (Scan) - O(N)
        # For small datasets, this is perfectly fine.
        scored_docs = []
        for doc in self.doc_store:
            score = self._cosine_similarity(q_vec, doc["vector"])
            scored_docs.append((score, doc["text"]))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        top_docs = scored_docs[:top_k]

        # Formulate Context
        context_str = "\n".join(
            [f"- {text} (Score: {score:.4f})" for score, text in top_docs]
        )

        # Mock Generation
        answer = f"Based on the context, here is the answer to '{question}'..."

        return {
            "result": answer,
            "source_documents": context_str,
            "top_matches": top_docs,
            "eval_score": self.calculate_faithfulness_score(answer, context_str),
        }

    def calculate_faithfulness_score(self, answer: str, context: str) -> float:
        """
        Evaluate: Quantify 'Hallucination' risk (Simple Overlap Metric).

        Logic:
        Calculates what percentage of significant words in the answer
        can be found in the retrieved context.

        metric = (Keywords in Answer found in Context) / (Total Keywords in Answer)

        Returns:
            float: 0.0 to 1.0 (1.0 means fully supported by context)
        """
        # Normalize
        ans_tokens = set(self._simple_tokenize(answer))
        ctx_tokens = set(self._simple_tokenize(context))

        # Filter stopwords (very basic list for demo)
        stopwords = {
            "the",
            "is",
            "a",
            "an",
            "and",
            "or",
            "of",
            "to",
            "in",
            "on",
            "it",
            "this",
            "that",
            "based",
            "here",
            "answer",
        }
        significant_tokens = {t for t in ans_tokens if t not in stopwords}

        if not significant_tokens:
            return 0.0

        supported_tokens = 0
        for token in significant_tokens:
            if token in ctx_tokens:
                supported_tokens += 1

        return round(supported_tokens / len(significant_tokens), 2)


if __name__ == "__main__":
    # Self-test if run directly
    rag = OptimizationRAG()
    rag.ingest_text("Python is great. Antigravity uses Windows. Coding is fun.")
    print(rag.query("What does Antigravity use?"))
