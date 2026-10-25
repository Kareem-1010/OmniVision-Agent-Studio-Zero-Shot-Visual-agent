"""
Resilient JSON Repair Engine and Fallback Parser.
Heuristically sanitizes malformed LLM outputs into compliant JSON structures.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger("OmniVision.JSONRepair")


def extract_fenced_json(text: str) -> str:
    """Extract content from ```json ... ``` or ``` ... ``` blocks if present."""
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def repair_json(text: str) -> str:
    """
    Attempt to repair common LLM JSON syntax errors:
    - Markdown code fences
    - Python literals (True, False, None)
    - Trailing commas in arrays and objects
    - Unquoted dictionary keys
    - Single quotes instead of double quotes
    - Missing closing brackets/braces due to token truncation
    """
    cleaned = extract_fenced_json(text).strip()
    if not cleaned:
        return "{}"

    # First, test if it is already valid JSON
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    # Extract outermost JSON boundary: find first '{' or '['
    start_brace = cleaned.find("{")
    start_bracket = cleaned.find("[")

    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        start_idx = start_brace
    elif start_bracket != -1:
        start_idx = start_bracket
    else:
        start_idx = 0

    cleaned = cleaned[start_idx:]

    # Convert Python boolean and None literals to JSON
    cleaned = re.sub(r"\bTrue\b", "true", cleaned)
    cleaned = re.sub(r"\bFalse\b", "false", cleaned)
    cleaned = re.sub(r"\bNone\b", "null", cleaned)

    # Convert single-quoted strings: 'val' -> "val"
    cleaned = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', cleaned)

    # Convert unquoted object keys: {name: "val"} -> {"name": "val"}
    cleaned = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', cleaned)

    # Remove trailing commas before closing braces or brackets: [1, 2, ] -> [1, 2]
    while True:
        subbed = re.sub(r",\s*([\]}])", r"\1", cleaned)
        if subbed == cleaned:
            break
        cleaned = subbed

    # Stack-based bracket/brace nesting tracker
    stack = []
    in_string = False
    escape = False

    for char in cleaned:
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if char in ("{", "["):
            stack.append(char)
        elif char == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif char == "]":
            if stack and stack[-1] == "[":
                stack.pop()

    # If inside an unclosed string, close it
    if in_string:
        cleaned += '"'

    # Close remaining brackets in exact reverse stack order
    closing_map = {"{": "}", "[": "]"}
    while stack:
        open_bracket = stack.pop()
        cleaned += closing_map[open_bracket]

    # Final pass removing any trailing commas created before closures
    while True:
        subbed = re.sub(r",\s*([\]}])", r"\1", cleaned)
        if subbed == cleaned:
            break
        cleaned = subbed

    return cleaned


def parse_json_safely(text: str, default: Optional[Any] = None) -> Any:
    """
    Parse JSON text with automatic multi-stage repair fallback.
    Returns parsed object or default value if all attempts fail.
    """
    if not text or not text.strip():
        return default if default is not None else {}

    # Stage 1: Direct JSON parsing
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Stage 2: Code block stripping + recursive repair
    try:
        repaired = repair_json(text)
        return json.loads(repaired)
    except Exception as e:
        logger.warning(f"JSON repair failed on input (len={len(text)}): {e}")

    # Stage 3: Aggressive line-based fallback if object
    try:
        lines = [line.strip() for line in text.splitlines() if ":" in line]
        if lines:
            constructed: dict = {}
            for line in lines:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    k = parts[0].strip().strip('"\'')
                    v = parts[1].strip().strip('"\',')
                    constructed[k] = v
            if constructed:
                return constructed
    except Exception:
        pass

    return default if default is not None else {}
