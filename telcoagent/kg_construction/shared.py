"""Helpers shared by the three KG-construction agents.

LLM-response JSON parsing (with best-effort repair of common LLM JSON
mistakes) and the inter-call rate-limit delay used by every agent.
"""

from __future__ import annotations

import ast
import json
import os
import re

# Rate-limit delay (seconds) between LLM calls for free-tier APIs
_RPM_DELAY = float(os.environ.get("TELCOAGENT_RPM_DELAY", "15"))

_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _repair_json(text: str) -> str:
    """Best-effort repair of common LLM JSON mistakes (trailing commas)."""
    return _TRAILING_COMMA_RE.sub(r"\1", text)


def _parse_json_response(text: str) -> list:
    """Extract JSON array from LLM response, handling markdown fences."""
    if not text or not text.strip():
        return []

    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end]).strip()

    if not text:
        return []

    # Extract JSON array substring
    arr_start = text.find("[")
    arr_end = text.rfind("]")
    candidate = text[arr_start : arr_end + 1] if (arr_start != -1 and arr_end > arr_start) else text

    # Attempt 1: strict JSON
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Attempt 2: repair trailing commas
    try:
        return json.loads(_repair_json(candidate))
    except json.JSONDecodeError:
        pass

    # Attempt 3: ast.literal_eval handles single-quoted keys/values
    try:
        result = ast.literal_eval(candidate)
        if isinstance(result, list):
            return result
    except Exception:
        pass

    raise json.JSONDecodeError("Could not parse JSON array", text, 0)
