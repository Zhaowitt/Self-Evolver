"""Shared text normalization for skill ids, content hashes, and markdown."""

from __future__ import annotations

import hashlib
import re


def slug(value: str) -> str:
    """Normalize free text into a stable snake_case identifier."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def normalize_content(content: str) -> str:
    """Whitespace/case-insensitive canonical form used for content hashing."""
    return "\n".join(line.strip() for line in content.lower().splitlines() if line.strip())


def content_hash(content: str) -> str:
    return hashlib.sha256(normalize_content(content).encode("utf-8")).hexdigest()
