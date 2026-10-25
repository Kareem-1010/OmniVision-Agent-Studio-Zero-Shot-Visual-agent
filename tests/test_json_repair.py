"""
Unit tests for JSON repair and fallback parser.
"""

from src.utils.json_repair import repair_json, parse_json_safely


def test_markdown_fence_removal():
    raw = "```json\n{\"intent\": \"count\", \"entity\": \"car\"}\n```"
    repaired = repair_json(raw)
    parsed = parse_json_safely(raw)
    assert parsed["intent"] == "count"
    assert parsed["entity"] == "car"


def test_single_quotes_and_python_literals():
    raw = "{'count': 5, 'detected': True, 'missing': None}"
    parsed = parse_json_safely(raw)
    assert parsed["count"] == 5
    assert parsed["detected"] is True
    assert parsed["missing"] is None


def test_unquoted_keys_and_trailing_commas():
    raw = "{intent: 'attribute', targets: ['red', 'car',], }"
    parsed = parse_json_safely(raw)
    assert parsed["intent"] == "attribute"
    assert parsed["targets"] == ["red", "car"]


def test_truncated_json_closure():
    raw = "{\"query\": \"find dogs\", \"nodes\": [{\"id\": 1, \"name\": \"dog\""
    parsed = parse_json_safely(raw)
    assert parsed["query"] == "find dogs"
    assert len(parsed["nodes"]) == 1
    assert parsed["nodes"][0]["name"] == "dog"
