"""Parsing utilities for controller JSON responses."""

from __future__ import annotations

import json
import re
from typing import Optional

from src.controller.schema import ControllerSignal


def extract_json_object(content: str) -> str:
    """Extract the first JSON object from a raw LLM response."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = content.find("{")
    if start < 0:
        raise ValueError("controller response did not contain a JSON object")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(content)):
        char = content[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : idx + 1]

    raise ValueError("controller response JSON object was incomplete")


def parse_controller_response(
    content: str,
    mode: str = "train",
    source: str = "llm",
) -> ControllerSignal:
    """Parse a controller response, returning a safe empty signal on failure."""
    try:
        json_text = extract_json_object(content or "")
        data = json.loads(json_text)
        if not isinstance(data, dict):
            raise ValueError("controller response JSON must be an object")
        return ControllerSignal.from_dict(
            data,
            mode=mode,
            source=source,
            raw_response=content,
        )
    except Exception as exc:
        return ControllerSignal.empty(
            mode=mode,
            source=source,
            raw_response=content,
            parse_error=str(exc),
        )


def parse_controller_dict(
    data: dict,
    mode: Optional[str] = None,
    source: str = "",
) -> ControllerSignal:
    """Validate a dict already loaded from JSON."""
    return ControllerSignal.from_dict(data, mode=mode, source=source)
