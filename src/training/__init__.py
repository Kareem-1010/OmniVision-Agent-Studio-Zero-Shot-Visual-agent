"""OmniVision Self-Improvement and Model Training Package.

Provides dataset building from agent execution traces (SFT/DPO),
Planner DPO optimization, teacher-to-student detector distillation,
and SigLIP/CLIP linear probing with temperature calibration.
"""

from src.training.dataset_builder import (
    AgentTrajectory,
    DatasetBuilder,
    DPOPreferencePair,
    SFTSample,
)
from src.training.distillation import (
    DistillationLossOutput,
    StudentDetectorDistiller,
    StudentDetectorNetwork,
)
from src.training.linear_probe import (
    LinearProbeHead,
    ProbeEpochMetrics,
    SigLIPLinearProbeTrainer,
    compute_ece,
)
from src.training.sft_dpo_pipeline import (
    LightweightPlannerPolicy,
    PlannerSelfImprovementTrainer,
    TrainingStepMetrics,
    compute_dpo_loss,
)

__all__ = [
    "AgentTrajectory",
    "SFTSample",
    "DPOPreferencePair",
    "DatasetBuilder",
    "compute_dpo_loss",
    "TrainingStepMetrics",
    "LightweightPlannerPolicy",
    "PlannerSelfImprovementTrainer",
    "DistillationLossOutput",
    "StudentDetectorNetwork",
    "StudentDetectorDistiller",
    "ProbeEpochMetrics",
    "LinearProbeHead",
    "compute_ece",
    "SigLIPLinearProbeTrainer",
]
