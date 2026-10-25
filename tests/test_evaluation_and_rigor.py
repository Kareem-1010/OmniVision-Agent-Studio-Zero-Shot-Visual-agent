"""
Tests for Phase 9: Evaluation & Scientific Research Rigor.
Verifies:
1. 300-sample multimodal benchmark builder and task taxonomy filters.
2. Metrics engine: VQA accuracy, counting MAE/RMSE, grounding IoU, MOTA/IDF1, tIoU, ECE, Brier score.
3. Statistical rigor: Bootstrap CIs, paired hypothesis tests (t-test, Wilcoxon), Cohen's d, McNemar test, Holm FDR.
4. Systematic ablation study: 10 canonical configs, degradation rankings, and significance vs Full Agent.
5. Scientific reporting: LaTeX tables, Markdown tables, reliability diagrams, failure taxonomy, research paper report.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

from src.eval.datasets import BenchmarkDataset, BenchmarkSample, Custom300BenchmarkBuilder
from src.eval.metrics import (
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
    adjust_p_values_fdr,
    compute_bootstrap_ci,
    compute_cohens_d,
    compute_mcnemar_test,
    paired_significance_test,
)
from src.eval.ablations import AblationConfig, AblationRunner
from src.eval.reporting import (
    classify_failure_cases,
    generate_calibration_plot,
    generate_latex_table,
    generate_markdown_table,
    generate_paper_report,
)


def test_benchmark_dataset_construction_and_filtering(tmp_path: Path):
    """Verify 300-query benchmark synthesis, task family balance, and JSONL persistence."""
    dataset = Custom300BenchmarkBuilder.build_dataset()
    assert "OmniVision" in dataset.dataset_name
    assert len(dataset.samples) == 300

    # Verify key reasoning task families
    families = {
        "count",
        "spatial_relation",
        "grounding_refcoco",
        "ocr_scene_text",
        "safety_hazard",
        "temporal_localization",
        "3d_metric_distance",
        "counterfactual_simulation",
    }
    present_families = {s.task_family for s in dataset.samples}
    assert families.issubset(present_families)

    # Test filtering
    count_samples = dataset.filter_by_task("count")
    assert len(count_samples) >= 30
    for s in count_samples:
        assert s.ground_truth_count is not None
        assert s.ground_truth_count > 0

    # Test JSONL export
    jsonl_path = tmp_path / "benchmark_300.jsonl"
    exported_count = dataset.to_jsonl(str(jsonl_path))
    assert exported_count == 300
    assert jsonl_path.exists()
    assert jsonl_path.stat().st_size > 5000


def test_scientific_metrics_computations():
    """Verify accuracy, counting MAE/RMSE, grounding IoU, MOTA, tIoU, and calibration metrics."""
    # 1. VQA Accuracy (Exact & Soft)
    preds = ["4 cars", "Stop Sign", "Safe clearance", "Unknown"]
    gts = ["4 cars", "stop sign", "Safe clearance maintained", "Forklift"]
    acc = compute_vqa_accuracy(preds, gts)
    assert acc >= 0.75

    # 2. Counting MAE and RMSE
    p_cnt = [4, 2, 7, 10]
    g_cnt = [4, 3, 5, 10]
    mae, rmse = compute_counting_mae_rmse(p_cnt, g_cnt)
    assert mae == 0.75
    assert rmse > 0.75

    # 3. Grounding Box IoU
    box_a = [0.1, 0.1, 0.5, 0.5]
    box_b = [0.1, 0.1, 0.5, 0.5]
    assert calculate_box_iou(box_a, box_b) == 1.0

    box_c = [0.6, 0.6, 0.9, 0.9]
    assert calculate_box_iou(box_a, box_c) == 0.0

    iou_at_50, miou = compute_grounding_iou([box_a, box_a], [box_b, box_c])
    assert iou_at_50 == 0.5
    assert miou == 0.5

    # 4. Tracking MOTA & IDF1
    mota, idf1 = compute_mota_idf1(
        false_positives=5,
        false_negatives=5,
        id_switches=2,
        ground_truth_objects=100,
        id_true_positives=88,
    )
    assert mota == 0.88
    assert idf1 > 0.85

    # 5. Temporal tIoU
    t_pred = [(2.0, 4.0), (10.0, 15.0)]
    t_gt = [(2.0, 4.0), (12.0, 15.0)]
    tiou = compute_temporal_tiou(t_pred, t_gt)
    assert 0.75 <= tiou <= 1.0

    # 6. Tool selection Jaccard accuracy
    p_tools = [["detector_ensemble", "critic"], ["open_vocab"]]
    e_tools = [["detector_ensemble", "critic"], ["open_vocab", "sam_segmentor"]]
    tool_acc = compute_tool_selection_accuracy(p_tools, e_tools)
    assert 0.70 <= tool_acc <= 0.80

    # 7. ECE and Brier Score
    confs = [0.95, 0.85, 0.75, 0.20]
    outcomes = [1, 1, 1, 0]
    ece, brier = compute_ece_and_brier(confs, outcomes)
    assert 0.0 <= ece <= 0.25
    assert 0.0 <= brier <= 0.15


def test_statistical_rigor_engine():
    """Verify bootstrap confidence intervals, paired hypothesis testing, Cohen's d, and McNemar test."""
    # 1. Bootstrap CI
    sample_scores = [0.85, 0.90, 0.88, 0.92, 0.87, 0.95, 0.89, 0.91]
    ci_low, ci_high = compute_bootstrap_ci(sample_scores, n_bootstraps=500, ci=0.95)
    assert 0.80 <= ci_low <= np.mean(sample_scores)
    assert np.mean(sample_scores) <= ci_high <= 1.0

    # 2. Cohen's d effect size
    base_dist = np.array([0.55, 0.58, 0.60, 0.52, 0.57])
    agent_dist = np.array([0.90, 0.92, 0.95, 0.88, 0.93])
    d = compute_cohens_d(base_dist, agent_dist)
    assert d > 2.0  # Very large positive effect

    # 3. Paired significance test
    sig_res = paired_significance_test(
        baseline_scores=base_dist.tolist(),
        agent_scores=agent_dist.tolist(),
    )
    assert sig_res.is_significant is True
    assert sig_res.p_value < 0.01
    assert sig_res.effect_magnitude == "large"
    assert sig_res.difference_mean > 0.30

    # 4. McNemar's Test
    base_hits = [True] * 5 + [False] * 10 + [False] * 5
    agent_hits = [True] * 5 + [True] * 10 + [False] * 5
    mcnemar = compute_mcnemar_test(base_hits, agent_hits)
    assert mcnemar["agent_wins"] == 10
    assert mcnemar["baseline_wins"] == 0
    assert mcnemar["is_significant"] is True

    # 5. Holm FDR adjustment
    p_vals = [0.001, 0.02, 0.04, 0.08]
    adj_p = adjust_p_values_fdr(p_vals, method="holm")
    assert len(adj_p) == 4
    assert adj_p[0] <= adj_p[1] <= adj_p[2] <= adj_p[3]


def test_ablation_study_suite():
    """Verify execution of the 10 canonical ablation configurations and ranking against full agent."""
    dataset = Custom300BenchmarkBuilder.build_dataset()
    # Evaluate across a subset of 30 samples for fast test execution
    sub_dataset = BenchmarkDataset(
        dataset_name="Sub-30-Benchmark",
        description="Sub-sample for fast ablation testing",
        samples=dataset.samples[:30],
    )

    configs = AblationRunner.get_canonical_10_configs()
    assert len(configs) == 10

    # Check that required configurations are present
    cfg_ids = {c.config_id for c in configs}
    assert "full_omnivision" in cfg_ids
    assert "monolithic_vlm" in cfg_ids
    assert "no_critic" in cfg_ids
    assert "no_active_perception" in cfg_ids
    assert "no_metric_depth" in cfg_ids

    results = AblationRunner.run_suite(sub_dataset, configs=configs)
    assert len(results) == 10

    full_res = results[0]
    assert full_res.config.config_id == "full_omnivision"
    assert full_res.metrics.overall_accuracy >= 0.80

    # Monolithic VLM should exhibit significant negative delta
    mono_res = next(r for r in results if r.config.config_id == "monolithic_vlm")
    assert mono_res.delta_accuracy < -0.20
    assert mono_res.significance_vs_full is not None
    assert mono_res.significance_vs_full.is_significant is True


def test_scientific_reporting_and_artifacts(tmp_path: Path):
    """Verify LaTeX table generator, Markdown table generator, calibration plot, and research report."""
    dataset = Custom300BenchmarkBuilder.build_dataset()
    sub_dataset = BenchmarkDataset(
        dataset_name="Sub-Reporting",
        description="Sub-sample",
        samples=dataset.samples[:20],
    )
    ablation_results = AblationRunner.run_suite(sub_dataset)

    # 1. LaTeX table formatting
    latex = generate_latex_table(ablation_results)
    assert "\\begin{table*}" in latex
    assert "\\toprule" in latex
    assert "\\bottomrule" in latex
    assert "\\end{table*}" in latex
    assert "Full OmniVision" in latex

    # 2. Markdown table formatting
    md_tbl = generate_markdown_table(ablation_results)
    assert "| Configuration | Accuracy (%) |" in md_tbl
    assert "Full OmniVision Agent" in md_tbl
    assert "Monolithic Single VLM" in md_tbl

    # 3. Calibration plot generation with matplotlib
    confidences = [0.95, 0.90, 0.80, 0.70, 0.60, 0.40, 0.30, 0.20]
    outcomes = [1, 1, 1, 1, 0, 0, 0, 0]
    plot_path = tmp_path / "calibration_reliability_diagram.png"
    saved_path = generate_calibration_plot(confidences, outcomes, str(plot_path))
    assert Path(saved_path).exists()
    assert Path(saved_path).stat().st_size > 1000

    # 4. Failure case classification taxonomy
    mock_preds = [
        {"answer": "2", "is_correct": False, "box_iou": 0.8},
        {"answer": "truck", "is_correct": False, "hallucination_detected": True},
        {"answer": "5 meters", "is_correct": False, "box_iou": 0.2},
    ]
    mock_gts = [
        {"query_id": "Q1", "task_family": "count", "ground_truth_answer": "4"},
        {"query_id": "Q2", "task_family": "safety_hazard", "ground_truth_answer": "spill"},
        {"query_id": "Q3", "task_family": "3d_metric_distance", "ground_truth_answer": "12 meters"},
    ]
    failures = classify_failure_cases(mock_preds, mock_gts)
    assert len(failures) == 3
    cats = {f["category"] for f in failures}
    assert "Perception Miss" in cats
    assert "Hallucination" in cats
    assert "Geometry Inaccuracy" in cats

    # 5. Scientific Paper Report Generator
    report_path = tmp_path / "research_evaluation_paper.md"
    paper_report = generate_paper_report(
        benchmark_results={"accuracy": 0.93},
        ablation_results=ablation_results,
        failure_analysis=failures,
        filepath=str(report_path),
    )
    assert "# OmniVision Agent: Research Evaluation & Scientific Rigor Report" in paper_report
    assert "Systematic Ablation Study Results" in paper_report
    assert Path(report_path).exists()
    assert Path(report_path).stat().st_size > 500
