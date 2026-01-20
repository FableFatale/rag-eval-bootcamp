import unittest

from rag_system.core import OptimizationRAG


class TestOptimizationRAG(unittest.TestCase):
    def setUp(self):
        """Set up a fresh RAG instance for each test."""
        self.rag = OptimizationRAG()

    def test_ingest_and_retrieval(self):
        """
        Test that we can ingest text and retrieve it.
        Integration style test.
        """
        text = "The sky is blue. The grass is green."
        self.rag.ingest_text(text)

        # We expect a query about 'sky' to retrieve the 'blue' sentence
        result = self.rag.query("sky")

        # Check if the top result contains the relevant text
        top_match_text = result["top_matches"][0][1]
        self.assertIn("The sky is blue", top_match_text)

    def test_empty_query(self):
        """Test how system handles empty state."""
        result = self.rag.query("anything")
        # Should execute without error, just empty results or low scores
        self.assertIsInstance(result, dict)

    def test_cosine_similarity(self):
        """Test the math logic directly."""
        # Identical vectors should have similarity ~1.0
        vec = [1.0, 0.0]
        score = self.rag._cosine_similarity(vec, vec)
        self.assertAlmostEqual(score, 1.0)

        # Orthogonal vectors should have similarity 0.0
        vec2 = [0.0, 1.0]
        score = self.rag._cosine_similarity(vec, vec2)
        self.assertAlmostEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
