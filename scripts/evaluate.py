from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_eval import BM25Retriever, hit_rate_at_k, mean_reciprocal_rank


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    corpus = load_json(ROOT / "examples" / "corpus.json")
    queries = load_json(ROOT / "examples" / "queries.json")
    retriever = BM25Retriever(corpus)

    rankings = [
        [doc_id for doc_id, _ in retriever.search(item["query"], top_k=3)]
        for item in queries
    ]
    expected = [item["expected_id"] for item in queries]

    print(f"queries={len(queries)}")
    print(f"hit_rate_at_3={hit_rate_at_k(rankings, expected, 3):.2f}")
    print(f"mean_reciprocal_rank={mean_reciprocal_rank(rankings, expected):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

