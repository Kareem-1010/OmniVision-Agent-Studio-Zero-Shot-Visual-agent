"""
Core Domain Preset Architecture and Registry.
Defines foundational configuration models and contracts for domain presets
that tailor the prompt, toolset, parameters, and post-processors to specific verticals.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel, Field

logger = logging.getLogger("OmniVision.Presets.Base")


class DomainPresetConfig(BaseModel):
    """Configuration specification for a specialized domain mode."""
    name: str = Field(description="Unique domain identifier (e.g. retail, accessibility, hazard_safety)")
    description: str = Field(description="High-level summary of the domain capabilities")
    system_prompt: str = Field(description="Domain-specialized system instruction for planner and synthesizer")
    enabled_tools: List[str] = Field(
        default_factory=list,
        description="Active subset of vision perception tools permitted for this domain",
    )
    confidence_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    max_iterations: int = Field(default=8, ge=1, le=24)
    budget_limit_usd: float = Field(default=0.30, ge=0.01)
    custom_parameters: Dict[str, Any] = Field(default_factory=dict)


class BaseDomainPreset(abc.ABC):
    """Abstract Base Class for vertical domain presets."""

    def __init__(self, config: Optional[DomainPresetConfig] = None) -> None:
        self._config = config or self._default_config()

    @property
    def config(self) -> DomainPresetConfig:
        return self._config

    @abc.abstractmethod
    def _default_config(self) -> DomainPresetConfig:
        """Construct default domain preset configuration."""
        raise NotImplementedError

    def get_system_prompt(self) -> str:
        """Retrieve specialized system instruction."""
        return self.config.system_prompt

    def get_toolset(self) -> List[str]:
        """Retrieve list of enabled perception tools."""
        return list(self.config.enabled_tools)

    @abc.abstractmethod
    def analyze(self, media_uri: str, query: Optional[str] = None, **kwargs: Any) -> Any:
        """Execute domain-specific analysis pipeline on image or video."""
        raise NotImplementedError


class PresetRegistry:
    """Central registry and factory for domain presets."""

    def __init__(self) -> None:
        self._presets: Dict[str, BaseDomainPreset] = {}

    def register(self, preset: BaseDomainPreset) -> None:
        """Register a domain preset instance."""
        name = preset.config.name.lower().strip()
        self._presets[name] = preset
        logger.info(f"Registered domain preset: '{name}'")

    def get(self, domain_name: str) -> BaseDomainPreset:
        """Retrieve a domain preset by name."""
        name = domain_name.lower().strip()
        if name not in self._presets:
            raise KeyError(f"Domain preset '{name}' not found. Available: {list(self._presets.keys())}")
        return self._presets[name]

    def list_presets(self) -> List[str]:
        """List all registered domain preset identifiers."""
        return list(self._presets.keys())


# Singleton instance
_GLOBAL_PRESET_REGISTRY = PresetRegistry()


def get_preset_registry() -> PresetRegistry:
    """Access the global preset registry."""
    return _GLOBAL_PRESET_REGISTRY
