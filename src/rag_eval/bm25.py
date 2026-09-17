from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re


TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Document:
    id: str
    text: str


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, documents: list[dict[str, str]], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("At least one document is required.")

        self.documents = [Document(id=item["id"], text=item["text"]) for item in documents]
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc.text) for doc in self.documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.term_frequencies = [Counter(tokens) for tokens in self.doc_tokens]
        self.document_frequencies = self._document_frequencies()

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        query_terms = tokenize(query)
        scores = [
            (document.id, self._score(query_terms, index))
            for index, document in enumerate(self.documents)
        ]
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def _score(self, query_terms: list[str], index: int) -> float:
        score = 0.0
        doc_length = self.doc_lengths[index] or 1
        frequencies = self.term_frequencies[index]

        for term in query_terms:
            if frequencies[term] == 0:
                continue
            idf = self._idf(term)
            numerator = frequencies[term] * (self.k1 + 1)
            denominator = frequencies[term] + self.k1 * (
                1 - self.b + self.b * doc_length / self.avg_doc_length
            )
            score += idf * numerator / denominator
        return score

    def _idf(self, term: str) -> float:
        total_docs = len(self.documents)
        docs_with_term = self.document_frequencies.get(term, 0)
        return math.log(1 + (total_docs - docs_with_term + 0.5) / (docs_with_term + 0.5))

    def _document_frequencies(self) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            frequencies.update(set(tokens))
        return frequencies

