"""
OmniVision Domain Presets and Demo Modes.
Provides configuration, prompts, toolset selection, and specialized analytics
for Retail, Accessibility, and Hazard/Safety domains.
"""

from src.presets.base import BaseDomainPreset, DomainPresetConfig, PresetRegistry, get_preset_registry
from src.presets.retail import RetailKPIReport, RetailPreset
from src.presets.accessibility import AccessibilityNarrationReport, AccessibilityPreset, TTSSynthesizer
from src.presets.hazard_safety import HazardSafetyPreset, SafetyAuditReport

__all__ = [
    "DomainPresetConfig",
    "BaseDomainPreset",
    "PresetRegistry",
    "get_preset_registry",
    "RetailKPIReport",
    "RetailPreset",
    "AccessibilityNarrationReport",
    "AccessibilityPreset",
    "TTSSynthesizer",
    "SafetyAuditReport",
    "HazardSafetyPreset",
]
