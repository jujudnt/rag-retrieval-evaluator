from __future__ import annotations


def hit_rate_at_k(rankings: list[list[str]], expected_ids: list[str], k: int) -> float:
    if len(rankings) != len(expected_ids):
        raise ValueError("Rankings and expected ids must have the same length.")
    hits = [expected in ranking[:k] for ranking, expected in zip(rankings, expected_ids)]
    return sum(hits) / len(hits) if hits else 0.0


def mean_reciprocal_rank(rankings: list[list[str]], expected_ids: list[str]) -> float:
    if len(rankings) != len(expected_ids):
        raise ValueError("Rankings and expected ids must have the same length.")

    reciprocal_ranks: list[float] = []
    for ranking, expected in zip(rankings, expected_ids):
        try:
            rank = ranking.index(expected) + 1
        except ValueError:
            rank = 0
        reciprocal_ranks.append(1 / rank if rank else 0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

