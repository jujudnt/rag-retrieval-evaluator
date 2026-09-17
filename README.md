# RAG Retrieval Evaluator

A tiny benchmark harness for retrieval-augmented generation systems.

It uses a pure-Python BM25 retriever and reports common retrieval metrics. The
project is intentionally small: recruiters can read the full implementation in a
few minutes, run the demo, and see how retrieval quality is measured before an
LLM is even called.

## Run

```bash
python3 scripts/evaluate.py
PYTHONPATH=src python3 -m unittest discover -s tests
```

Example output:

```text
queries=4
hit_rate_at_3=1.00
mean_reciprocal_rank=0.88
```

## What It Shows

- BM25 scoring
- top-k retrieval
- hit rate at k
- mean reciprocal rank
- JSON data contracts for corpus and queries
