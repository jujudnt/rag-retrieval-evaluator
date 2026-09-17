import unittest

from rag_eval import BM25Retriever, hit_rate_at_k, mean_reciprocal_rank


class RetrievalTest(unittest.TestCase):
    def test_returns_best_matching_document(self):
        retriever = BM25Retriever(
            [
                {"id": "a", "text": "billing invoice refund"},
                {"id": "b", "text": "security sso audit logs"},
            ]
        )

        ranking = retriever.search("invoice refund", top_k=1)

        self.assertEqual(ranking[0][0], "a")

    def test_metrics(self):
        rankings = [["a", "b"], ["c", "d"]]
        expected = ["a", "d"]

        self.assertEqual(hit_rate_at_k(rankings, expected, 2), 1.0)
        self.assertEqual(mean_reciprocal_rank(rankings, expected), 0.75)


if __name__ == "__main__":
    unittest.main()

