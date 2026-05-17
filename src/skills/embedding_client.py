"""Optional embedding similarity helpers for skill deduplication."""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Iterable, List, Optional


class EmbeddingClient:
    """Small embedding interface used by the skill deduper."""

    def __init__(self, embed_fn: Optional[Callable[[str], List[float]]] = None):
        self.embed_fn = embed_fn

    def embed(self, text: str) -> Optional[List[float]]:
        if not self.embed_fn:
            return None
        return self.embed_fn(text)

    def similarity(self, left: str, right: str) -> Optional[float]:
        left_vec = self.embed(left)
        right_vec = self.embed(right)
        if left_vec is None or right_vec is None:
            return None
        return cosine_similarity(left_vec, right_vec)


def local_text_similarity(left: str, right: str) -> float:
    """Offline fallback combining token Jaccard and bag-of-words cosine."""
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    jaccard = len(set(left_tokens) & set(right_tokens)) / len(set(left_tokens) | set(right_tokens))
    cosine = _counter_cosine(Counter(left_tokens), Counter(right_tokens))
    return max(jaccard, cosine)


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values) or not left_values:
        return 0.0
    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(a * a for a in left_values))
    right_norm = math.sqrt(sum(b * b for b in right_values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _counter_cosine(left: Counter[str], right: Counter[str]) -> float:
    terms = set(left) | set(right)
    dot = sum(left[term] * right[term] for term in terms)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())
