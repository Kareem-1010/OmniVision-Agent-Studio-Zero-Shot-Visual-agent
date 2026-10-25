"""
Unit tests for Prompt Registry.
"""

import pytest
from src.utils.prompt_registry import PromptRegistry


def test_prompt_registration_and_rendering():
    registry = PromptRegistry()
    registry.register(
        name="test_prompt",
        version="1.0.0",
        template="Hello {name}, your task is {task}.",
        description="Test greeting template",
    )

    rendered = registry.render("test_prompt", name="Agent", task="perception")
    assert rendered == "Hello Agent, your task is perception."


def test_versioning_and_rollback():
    registry = PromptRegistry()
    registry.register(
        name="task_spec",
        version="1.0.0",
        template="V1: {input}",
        set_active=True,
    )
    registry.register(
        name="task_spec",
        version="2.0.0",
        template="V2: {input}",
        set_active=True,
    )

    # By default active version should be 2.0.0
    assert registry.render("task_spec", input="data") == "V2: data"

    # Explicit version lookup
    assert registry.render("task_spec", version="1.0.0", input="data") == "V1: data"


def test_missing_variable_validation():
    registry = PromptRegistry()
    registry.register(
        name="strict_prompt",
        version="1.0.0",
        template="Execute {action} on {target}",
    )
    with pytest.raises(ValueError):
        registry.render("strict_prompt", action="detect")
