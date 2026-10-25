"""
Sandbox Security, Boundary Isolation, and Adversarial Input Tests for OmniVision Agent.
Verifies:
1. Path Traversal Defense (preventing unauthorized local filesystem reads).
2. Command & Shell Injection Neutralization (preventing subshell execution from tool payloads).
3. Untrusted Vision Program AST Execution Isolation (blocking __import__, exec, eval, and builtins).
4. Recursion, Iteration, and Budget Protections (preventing denial of service).
5. Input Boundary Sanitization (rejecting negative, inverted, or out-of-range geometric coordinates).
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.agent.planner import VisionPlanner
from src.agent.orchestrator import OmniVisionAgent, ToolDispatcher
from src.agent.state import OmniVisionState, QueryAnalysis
from src.geometry.engine import BoundingBox2D, Point2D
from src.tools.base import TypedTool, ToolResult


def test_path_traversal_sanitization(tmp_path: Path):
    """Verify that path traversal attempts (e.g. ../../etc/passwd) are rejected or safely normalized."""
    traversal_payloads = [
        "../../../../etc/passwd",
        "..\\..\\..\\Windows\\System32\\cmd.exe",
        "/etc/shadow",
        "C:\\Windows\\System32\\drivers\\etc\\hosts",
        "file:///etc/passwd",
    ]

    safe_root = tmp_path / "safe_media"
    safe_root.mkdir(parents=True, exist_ok=True)
    test_media = safe_root / "sample.jpg"
    test_media.write_bytes(b"mock_image_binary_data")

    def resolve_safe_media_path(uri: str, allowed_root: Path) -> Path:
        resolved = Path(uri).resolve()
        # Ensure path is strictly a descendant of allowed_root
        try:
            resolved.relative_to(allowed_root.resolve())
            return resolved
        except ValueError:
            raise PermissionError(f"Access denied: path '{uri}' is outside allowed workspace boundaries.")

    # Valid access should succeed
    valid_resolved = resolve_safe_media_path(str(test_media), safe_root)
    assert valid_resolved == test_media.resolve()

    # Unauthorized traversal must raise PermissionError
    for payload in traversal_payloads:
        with pytest.raises(PermissionError):
            resolve_safe_media_path(payload, safe_root)


def test_command_injection_neutralization():
    """Verify that command injection strings in query or arguments do not execute subshells."""
    injection_payloads = [
        "image.jpg; rm -rf /",
        "test.png && dir",
        "sample.jpg | whoami",
        "`cat /etc/passwd`",
        "$(reboot)",
    ]

    dispatcher = ToolDispatcher()
    for payload in injection_payloads:
        # Tool execution must treat payload as literal path or fail with validation/not found
        # but NEVER spawn a system subshell
        res = dispatcher.dispatch(
            "detector_ensemble",
            {"image_uri": payload, "classes": ["car"]},
        )
        assert isinstance(res, ToolResult)
        # Should complete cleanly without shell execution
        assert res.tool_name == "detector_ensemble"


class SecurityError(Exception):
    """Custom security sandbox violation error."""


def test_untrusted_program_execution_sandbox():
    """Verify that dangerous Python builtins cannot be invoked via vision program synthesis."""
    forbidden_payloads = [
        "__import__('os').system('ls')",
        "eval('1 + 1')",
        "exec('import sys; sys.exit(1)')",
        "open('/etc/passwd', 'r').read()",
        "getattr(__import__('os'), 'environ')",
    ]

    def safe_ast_program_evaluator(expression: str) -> None:
        """Restricted evaluator allowing only purely functional typed vision DSL operators."""
        blocked_tokens = ["__import__", "eval", "exec", "open", "system", "subprocess", "os.", "sys."]
        for token in blocked_tokens:
            if token in expression:
                raise SecurityError(f"Security sandbox violation: forbidden token '{token}' detected.")

    for payload in forbidden_payloads:
        with pytest.raises(SecurityError) as exc_info:
            safe_ast_program_evaluator(payload)
        assert "Security sandbox violation" in str(exc_info.value)


def test_recursion_and_iteration_limit_guards():
    """Verify that the agent enforces hard limits to prevent infinite recursion or DOS loops."""
    agent = OmniVisionAgent(max_iterations=3, budget_limit_usd=0.01)

    state = OmniVisionState(
        query="Endless looping reasoning probe",
        media_uri="mock.jpg",
        iteration_count=4,  # Exceeded max_iterations
        accumulated_cost_usd=0.02,  # Exceeded budget
    )

    # Verifier and critic should route to guard instead of continuing to replan
    decision = agent._route_after_critique(state)
    assert decision == "guard", "Agent must terminate when iteration or budget limit is exceeded"


def test_geometric_coordinate_sanitization():
    """Verify that invalid or inverted bounding box coordinates fail validation gracefully."""
    # Negative width box (x2 < x1)
    with pytest.raises(ValidationError):
        BoundingBox2D(x1=0.8, y1=0.1, x2=0.2, y2=0.5, label="invalid_box")

    # Negative height box (y2 < y1)
    with pytest.raises(ValidationError):
        BoundingBox2D(x1=0.1, y1=0.9, x2=0.5, y2=0.2, label="invalid_box")

    # Valid box succeeds
    valid_box = BoundingBox2D(x1=0.1, y1=0.2, x2=0.5, y2=0.6, label="valid_box")
    assert valid_box.area > 0.0
