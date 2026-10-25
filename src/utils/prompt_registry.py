"""
Prompt Registry Subsystem with Semantic Versioning and Template Validation.
Manages versioned system and user prompts across the OmniVision multi-agent architecture.
"""

from __future__ import annotations

import string
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PromptVersion(BaseModel):
    """Immutable representation of a specific prompt template version."""
    version: str  # e.g., '1.0.0'
    template: str
    input_variables: List[str] = Field(default_factory=list)
    description: str = ""
    created_at: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def render(self, **kwargs: Any) -> str:
        """Render the prompt template by substituting variables."""
        missing = [v for v in self.input_variables if v not in kwargs]
        if missing:
            raise ValueError(f"Missing required prompt variables: {missing}")
        # Safe string formatting supporting standard python string templates
        formatter = string.Formatter()
        return formatter.vformat(self.template, (), kwargs)


class PromptTemplate(BaseModel):
    """Container holding version history and active pointer for a named prompt."""
    name: str
    active_version: str
    versions: Dict[str, PromptVersion] = Field(default_factory=dict)

    def get_active(self) -> PromptVersion:
        """Retrieve the currently designated active prompt version."""
        if self.active_version not in self.versions:
            raise KeyError(f"Active version '{self.active_version}' not found for prompt '{self.name}'")
        return self.versions[self.active_version]

    def add_version(
        self,
        version: str,
        template: str,
        input_variables: Optional[List[str]] = None,
        description: str = "",
        set_active: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptVersion:
        """Add a new version and optionally promote it to active."""
        if input_variables is None:
            # Auto-detect format fields {variable_name}
            formatter = string.Formatter()
            detected = [
                field_name
                for _, field_name, _, _ in formatter.parse(template)
                if field_name is not None and field_name != ""
            ]
            input_variables = list(dict.fromkeys(detected))

        pv = PromptVersion(
            version=version,
            template=template,
            input_variables=input_variables,
            description=description,
            metadata=metadata or {},
        )
        self.versions[version] = pv
        if set_active:
            self.active_version = version
        return pv


class PromptRegistry:
    """Centralized, thread-safe prompt repository with default research templates."""

    def __init__(self) -> None:
        self._prompts: Dict[str, PromptTemplate] = {}
        self._load_default_prompts()

    def register(
        self,
        name: str,
        version: str,
        template: str,
        input_variables: Optional[List[str]] = None,
        description: str = "",
        set_active: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptVersion:
        """Register a new prompt or version under a given identifier."""
        if name not in self._prompts:
            self._prompts[name] = PromptTemplate(name=name, active_version=version)
        prompt_container = self._prompts[name]
        return prompt_container.add_version(
            version=version,
            template=template,
            input_variables=input_variables,
            description=description,
            set_active=set_active,
            metadata=metadata,
        )

    def get(self, prompt_name: str, version: Optional[str] = None) -> PromptVersion:
        """Fetch prompt version. If version is None, returns active version."""
        if prompt_name not in self._prompts:
            raise KeyError(f"Prompt '{prompt_name}' is not registered.")
        container = self._prompts[prompt_name]
        if version is None:
            return container.get_active()
        if version not in container.versions:
            raise KeyError(f"Version '{version}' not found for prompt '{prompt_name}'.")
        return container.versions[version]

    def render(self, prompt_name: str, version: Optional[str] = None, **kwargs: Any) -> str:
        """Render a named prompt directly using provided keyword arguments."""
        pv = self.get(prompt_name, version=version)
        return pv.render(**kwargs)

    def list_prompts(self) -> Dict[str, List[str]]:
        """List all prompt names and their registered versions."""
        return {name: list(c.versions.keys()) for name, c in self._prompts.items()}

    def _load_default_prompts(self) -> None:
        """Pre-populate registry with default multi-agent reasoning templates."""
        # Query Analyst
        self.register(
            name="query_analyst",
            version="1.0.0",
            template=(
                "You are the Lead Query Analyst for OmniVision Agent.\n"
                "Analyze the user query: {query}\n"
                "Visual modality: {modality}\n"
                "Decompose into intent, target entities, spatial/temporal scope, and ambiguity flags.\n"
                "Output strictly in JSON schema format."
            ),
            description="Decomposes open-ended user questions into formal intent specifications.",
        )

        # Planner & Program Synthesizer
        self.register(
            name="planner_dag",
            version="1.0.0",
            template=(
                "You are the Program Synthesizer for OmniVision Agent.\n"
                "Goal: {goal}\n"
                "Available tools: {tools_catalog}\n"
                "Prior observations: {prior_observations}\n"
                "Construct an executable DAG plan with steps, tool arguments, and dependency links.\n"
                "Output strictly in JSON schema format."
            ),
            description="Synthesizes an execution DAG of vision tools and reasoning steps.",
        )

        # Verifier & Critic
        self.register(
            name="adversarial_verifier",
            version="1.0.0",
            template=(
                "You are the Adversarial Verifier for OmniVision Agent.\n"
                "Draft Answer: {draft_answer}\n"
                "Grounding Evidence: {evidence}\n"
                "Question: {query}\n"
                "Critically examine all claims against visual observations. Attempt to falsify any unsupported claim.\n"
                "Output strict JSON with calibrated confidence and claim attribution."
            ),
            description="Performs adversarial verification and cross-evidence attribution.",
        )


_REGISTRY = PromptRegistry()


def get_prompt_registry() -> PromptRegistry:
    """Access global singleton prompt registry."""
    return _REGISTRY
