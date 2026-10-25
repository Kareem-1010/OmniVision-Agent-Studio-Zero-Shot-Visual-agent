"""OmniVision Benchmarks & Scientific Evaluation Package."""

from src.eval.datasets import (
    BenchmarkDataset,
    BenchmarkSample,
    Custom300BenchmarkBuilder,
)
from src.eval.metrics import (
    EvaluationMetricsSummary,
    calculate_box_iou,
    compute_counting_mae_rmse,
    compute_ece_and_brier,
    compute_grounding_iou,
    compute_mota_idf1,
    compute_temporal_tiou,
    compute_tool_selection_accuracy,
    compute_vqa_accuracy,
)
from src.eval.statistics import (
    SignificanceResult,
    adjust_p_values_fdr,
    classify_effect_magnitude,
    compute_bootstrap_ci,
    compute_cohens_d,
    compute_mcnemar_test,
    paired_significance_test,
)
from src.eval.ablations import (
    AblationConfig,
    AblationRunResult,
    AblationRunner,
)
from src.eval.reporting import (
    classify_failure_cases,
    generate_calibration_plot,
    generate_latex_table,
    generate_markdown_table,
    generate_paper_report,
)

__all__ = [
    # Datasets
    "BenchmarkSample",
    "BenchmarkDataset",
    "Custom300BenchmarkBuilder",
    # Metrics
    "EvaluationMetricsSummary",
    "calculate_box_iou",
    "compute_counting_mae_rmse",
    "compute_ece_and_brier",
    "compute_grounding_iou",
    "compute_mota_idf1",
    "compute_temporal_tiou",
    "compute_tool_selection_accuracy",
    "compute_vqa_accuracy",
    # Statistics
    "SignificanceResult",
    "adjust_p_values_fdr",
    "classify_effect_magnitude",
    "compute_bootstrap_ci",
    "compute_cohens_d",
    "compute_mcnemar_test",
    "paired_significance_test",
    # Ablations
    "AblationConfig",
    "AblationRunResult",
    "AblationRunner",
    # Reporting
    "classify_failure_cases",
    "generate_calibration_plot",
    "generate_latex_table",
    "generate_markdown_table",
    "generate_paper_report",
]
