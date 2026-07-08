"""Embedding similarity via an OpenAI-compatible endpoint, with lexical fallback helpers."""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from typing import Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Embeddings from an injected function or an OpenAI-compatible /embeddings endpoint.

    Configure the endpoint via env: EMBEDDING_MODEL (required), EMBEDDING_BASE_URL
    (falls back to OPENAI_BASE_URL), EMBEDDING_API_KEY (falls back to OPENAI_API_KEY).
    When embeddings are unavailable, methods return None so callers can fall back to
    `local_text_similarity`.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        model: str = "",
        base_url: Optional[str] = None,
        api_key: str = "",
    ):
        self.embed_fn = embed_fn
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._client = None

    @classmethod
    def from_env(cls) -> Optional["EmbeddingClient"]:
        """Build a client from EMBEDDING_* env vars, or None when unconfigured."""
        model = os.getenv("EMBEDDING_MODEL", "").strip()
        if not model:
            return None
        return cls(
            model=model,
            base_url=os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None,
            api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
        )

    def embed_many(self, texts: Iterable[str]) -> Optional[List[List[float]]]:
        """Embed a batch of texts; None means embeddings are unavailable."""
        text_list = list(texts)
        if not text_list:
            return []
        if self.embed_fn:
            return [list(self.embed_fn(text)) for text in text_list]
        if not self.model:
            return None
        try:
            if self._client is None:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = self._client.embeddings.create(model=self.model, input=text_list)
            items = sorted(response.data, key=lambda item: item.index)
            return [list(item.embedding) for item in items]
        except Exception as exc:
            logger.warning(f"Embedding request failed ({exc}); falling back to lexical similarity")
            return None

    def embed(self, text: str) -> Optional[List[float]]:
        vectors = self.embed_many([text])
        return vectors[0] if vectors else None

    def similarity(self, left: str, right: str) -> Optional[float]:
        vectors = self.embed_many([left, right])
        if not vectors:
            return None
        return cosine_similarity(vectors[0], vectors[1])


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
    return re.findall(r"[a-z0-9]+", text.lower())
