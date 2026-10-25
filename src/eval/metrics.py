"""
Scientific Metrics Engine for Multimodal Visual Reasoning.
Implements:
1. Exact and Soft VQA Accuracy.
2. Counting MAE and RMSE.
3. Grounding Spatial IoU@0.5 and mean IoU (mIoU).
4. Multi-Object Tracking MOTA and IDF1.
5. Temporal Grounding Interval tIoU.
6. Tool-Selection Accuracy.
7. Hallucination Rate (POPE-style False Claims / Total Claims).
8. Expected Calibration Error (ECE) and Brier Score.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field


class EvaluationMetricsSummary(BaseModel):
    """Aggregated quantitative evaluation metrics across a benchmark suite."""
    sample_count: int
    overall_accuracy: float = Field(ge=0.0, le=1.0)
    counting_mae: float = Field(ge=0.0)
    counting_rmse: float = Field(ge=0.0)
    grounding_iou_at_05: float = Field(ge=0.0, le=1.0)
    grounding_mean_iou: float = Field(ge=0.0, le=1.0)
    tracking_mota: float
    tracking_idf1: float = Field(ge=0.0, le=1.0)
    temporal_mean_tiou: float = Field(ge=0.0, le=1.0)
    tool_selection_accuracy: float = Field(ge=0.0, le=1.0)
    hallucination_rate: float = Field(ge=0.0, le=1.0)
    expected_calibration_error: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0, le=1.0)
    mean_plan_steps: float = Field(ge=0.0)
    mean_latency_ms: float = Field(ge=0.0)
    mean_token_cost_usd: float = Field(ge=0.0)


def compute_vqa_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compute exact and normalized keyword match VQA accuracy.
    """
    if not predictions:
        return 0.0

    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        p_clean = pred.lower().strip()
        g_clean = gt.lower().strip()

        # Direct equality or keyword containment
        if p_clean == g_clean or g_clean in p_clean:
            correct += 1
        else:
            # Word overlap threshold (e.g. >= 60% token overlap)
            p_words = set(p_clean.split())
            g_words = set(g_clean.split())
            if len(g_words) > 0 and len(p_words.intersection(g_words)) / len(g_words) >= 0.6:
                correct += 1

    return round(correct / len(predictions), 4)


def compute_counting_mae_rmse(
    predicted_counts: List[int],
    ground_truth_counts: List[int],
) -> Tuple[float, float]:
    """
    Compute Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for object counting.
    """
    if not predicted_counts:
        return 0.0, 0.0

    diffs = [abs(p - g) for p, g in zip(predicted_counts, ground_truth_counts)]
    sq_diffs = [(p - g) ** 2 for p, g in zip(predicted_counts, ground_truth_counts)]

    mae = float(np.mean(diffs))
    rmse = float(math.sqrt(np.mean(sq_diffs)))
    return round(mae, 3), round(rmse, 3)


def calculate_box_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection-over-Union (IoU) between two boxes [x1, y1, x2, y2]."""
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def compute_grounding_iou(
    pred_boxes_list: List[List[float]],
    gt_boxes_list: List[List[float]],
    threshold: float = 0.50,
) -> Tuple[float, float]:
    """
    Compute Grounding IoU@0.5 success rate and mean IoU (mIoU).
    """
    if not pred_boxes_list:
        return 0.0, 0.0

    ious: List[float] = []
    hits = 0

    for p_box, g_box in zip(pred_boxes_list, gt_boxes_list):
        iou = calculate_box_iou(p_box, g_box)
        ious.append(iou)
        if iou >= threshold:
            hits += 1

    iou_at_thresh = hits / max(1, len(pred_boxes_list))
    miou = float(np.mean(ious)) if ious else 0.0
    return round(iou_at_thresh, 4), round(miou, 4)


def compute_mota_idf1(
    false_positives: int,
    false_negatives: int,
    id_switches: int,
    ground_truth_objects: int,
    id_true_positives: int,
) -> Tuple[float, float]:
    """
    Compute MOTA (Multiple Object Tracking Accuracy) and IDF1 (ID F1-score).
    MOTA = 1 - (FN + FP + IDSW) / GT
    IDF1 = 2*IDTP / (2*IDTP + FP + FN)
    """
    gt = max(1, ground_truth_objects)
    mota = 1.0 - ((false_negatives + false_positives + id_switches) / float(gt))

    denom = 2 * id_true_positives + false_positives + false_negatives
    idf1 = (2 * id_true_positives) / float(denom) if denom > 0 else 0.0

    return round(mota, 4), round(idf1, 4)


def compute_temporal_tiou(
    pred_intervals: List[Tuple[float, float]],
    gt_intervals: List[Tuple[float, float]],
) -> float:
    """
    Compute mean temporal Intersection-over-Union (tIoU) for video intervals [s, e].
    """
    if not pred_intervals:
        return 0.0

    tious: List[float] = []
    for (s_p, e_p), (s_g, e_g) in zip(pred_intervals, gt_intervals):
        inter_start = max(s_p, s_g)
        inter_end = min(e_p, e_g)
        inter = max(0.0, inter_end - inter_start)

        union_start = min(s_p, s_g)
        union_end = max(e_p, e_g)
        union = max(0.001, union_end - union_start)

        tious.append(inter / union)

    return round(float(np.mean(tious)), 4)


def compute_tool_selection_accuracy(
    pred_tool_sets: List[List[str]],
    expected_tool_sets: List[List[str]],
) -> float:
    """
    Compute Jaccard similarity between activated tools and required ground truth toolset.
    """
    if not pred_tool_sets:
        return 0.0

    jaccards: List[float] = []
    for p_tools, e_tools in zip(pred_tool_sets, expected_tool_sets):
        p_set = set(p_tools)
        e_set = set(e_tools)
        union = p_set.union(e_set)
        if not union:
            jaccards.append(1.0)
        else:
            inter = p_set.intersection(e_set)
            jaccards.append(len(inter) / float(len(union)))

    return round(float(np.mean(jaccards)), 4)


def compute_ece_and_brier(
    confidences: List[float],
    binary_outcomes: List[int],
    n_bins: int = 10,
) -> Tuple[float, float]:
    """
    Compute Expected Calibration Error (ECE) and Brier Score.
    Brier = mean((conf - outcome)^2)
    """
    if not confidences:
        return 0.0, 0.0

    confs = np.array(confidences, dtype=np.float32)
    outs = np.array(binary_outcomes, dtype=np.float32)

    # 1. Brier Score
    brier = float(np.mean((confs - outs) ** 2))

    # 2. Expected Calibration Error
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(confs)

    for i in range(n_bins):
        b_low, b_high = bin_edges[i], bin_edges[i + 1]
        in_bin = (confs > b_low) & (confs <= b_high)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            avg_conf = float(np.mean(confs[in_bin]))
            avg_acc = float(np.mean(outs[in_bin]))
            weight = bin_count / float(total)
            ece += abs(avg_acc - avg_conf) * weight

    return round(float(ece), 4), round(brier, 4)
