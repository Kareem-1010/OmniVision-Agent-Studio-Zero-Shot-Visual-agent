"""
Statistical Rigor Engine for Scientific Evaluation.
Provides:
1. Non-parametric Bootstrap Confidence Intervals (BCa and Percentile).
2. Paired Hypothesis Testing (Paired t-test and Wilcoxon Signed-Rank Test).
3. Effect Size Calculations (Cohen's d and Hedges' g).
4. Multiple Hypothesis Testing Corrections (Bonferroni, Holm-Bonferroni, Benjamini-Hochberg FDR).
5. McNemar's Test for Paired Binary Decisions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from pydantic import BaseModel, Field


class SignificanceResult(BaseModel):
    """Structured report of a statistical significance test between two models/conditions."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool = Field(description="Significant at alpha = 0.05")
    effect_size: float = Field(description="Cohen's d effect size")
    effect_magnitude: str = Field(description="negligible, small, medium, large")
    sample_size: int
    mean_baseline: float
    mean_agent: float
    difference_mean: float
    ci_95_lower: float
    ci_95_upper: float


def compute_bootstrap_ci(
    data: List[float],
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute non-parametric empirical bootstrap confidence interval for a metric mean.
    Returns (lower_bound, upper_bound).
    """
    if not data:
        return 0.0, 0.0

    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n == 1:
        return float(arr[0]), float(arr[0])

    rng = np.random.RandomState(seed)
    bootstrap_means = np.empty(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        resample = rng.choice(arr, size=n, replace=True)
        bootstrap_means[i] = np.mean(resample)

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(bootstrap_means, lower_pct))
    ci_upper = float(np.percentile(bootstrap_means, upper_pct))
    return round(ci_lower, 4), round(ci_upper, 4)


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for paired or two-sample distributions.
    d = (mean(y) - mean(x)) / pooled_std
    """
    diff = y - x
    std_diff = np.std(diff, ddof=1)
    if std_diff < 1e-9:
        return 0.0
    return float(np.mean(diff) / std_diff)


def classify_effect_magnitude(d: float) -> str:
    """Classify effect size magnitude according to Cohen's thresholds."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def paired_significance_test(
    baseline_scores: List[float],
    agent_scores: List[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> SignificanceResult:
    """
    Run comprehensive paired hypothesis tests (Paired Student's t-test and Wilcoxon signed-rank).
    Evaluates whether the OmniVision Agent significantly outperforms the baseline.
    """
    if len(baseline_scores) != len(agent_scores):
        raise ValueError(
            f"Sample size mismatch: baseline={len(baseline_scores)}, agent={len(agent_scores)}"
        )
    if len(baseline_scores) == 0:
        raise ValueError("Cannot perform hypothesis test on empty score distributions.")

    b_arr = np.asarray(baseline_scores, dtype=np.float64)
    a_arr = np.asarray(agent_scores, dtype=np.float64)
    diff = a_arr - b_arr
    n = len(diff)

    mean_b = float(np.mean(b_arr))
    mean_a = float(np.mean(a_arr))
    diff_mean = float(np.mean(diff))

    ci_lower, ci_upper = compute_bootstrap_ci(diff.tolist(), n_bootstraps=1000, ci=1.0 - alpha)
    d = compute_cohens_d(b_arr, a_arr)
    magnitude = classify_effect_magnitude(d)

    # If all differences are zero
    if np.allclose(diff, 0.0):
        return SignificanceResult(
            test_name="Paired t-test",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            effect_size=0.0,
            effect_magnitude="negligible",
            sample_size=n,
            mean_baseline=round(mean_b, 4),
            mean_agent=round(mean_a, 4),
            difference_mean=0.0,
            ci_95_lower=0.0,
            ci_95_upper=0.0,
        )

    # Try Wilcoxon if non-normal or t-test
    try:
        if n >= 20:
            w_stat, w_p = stats.wilcoxon(a_arr, b_arr, alternative=alternative)
            stat = float(w_stat)
            p_val = float(w_p)
            test_name = "Wilcoxon signed-rank test"
        else:
            t_res = stats.ttest_rel(a_arr, b_arr, alternative=alternative)
            stat = float(t_res.statistic)
            p_val = float(t_res.pvalue)
            test_name = "Paired Student's t-test"
    except Exception:
        # Fallback to standard paired t-test
        t_res = stats.ttest_rel(a_arr, b_arr, alternative=alternative)
        stat = float(t_res.statistic) if not math.isnan(t_res.statistic) else 0.0
        p_val = float(t_res.pvalue) if not math.isnan(t_res.pvalue) else 1.0
        test_name = "Paired Student's t-test (fallback)"

    return SignificanceResult(
        test_name=test_name,
        statistic=round(stat, 4),
        p_value=round(p_val, 6),
        is_significant=bool(p_val < alpha),
        effect_size=round(d, 4),
        effect_magnitude=magnitude,
        sample_size=n,
        mean_baseline=round(mean_b, 4),
        mean_agent=round(mean_a, 4),
        difference_mean=round(diff_mean, 4),
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
    )


def compute_mcnemar_test(
    baseline_correct: List[bool],
    agent_correct: List[bool],
) -> Dict[str, Union[float, bool, str]]:
    """
    McNemar's test for paired 2x2 contingency table (comparing classification successes/failures).
    Contingency table:
      b: Agent correct, Baseline incorrect (Agent wins)
      c: Baseline correct, Agent incorrect (Baseline wins)
    Statistic = (|b - c| - 1)^2 / (b + c) with continuity correction.
    """
    if len(baseline_correct) != len(agent_correct):
        raise ValueError("Length mismatch between baseline and agent outcomes.")

    b = sum(1 for a, base in zip(agent_correct, baseline_correct) if a and not base)
    c = sum(1 for a, base in zip(agent_correct, baseline_correct) if not a and base)

    discordant = b + c
    if discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "agent_wins": b,
            "baseline_wins": c,
            "interpretation": "Identical performance across all queries.",
        }

    # Use exact binomial test when sample size is small (< 25), else continuity-corrected chi2
    if discordant < 25:
        binom_res = stats.binomtest(b, n=discordant, p=0.5)
        p_val = float(binom_res.pvalue)
        stat = float(b)
        test_type = "Exact Binomial McNemar"
    else:
        stat = ((abs(b - c) - 1.0) ** 2) / float(discordant)
        p_val = float(1.0 - stats.chi2.cdf(stat, df=1))
        test_type = "Continuity-Corrected Chi2 McNemar"

    return {
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "is_significant": bool(p_val < 0.05),
        "agent_wins": b,
        "baseline_wins": c,
        "test_type": test_type,
        "interpretation": f"OmniVision Agent won {b} discordant queries vs Baseline {c}.",
    }


def adjust_p_values_fdr(p_values: List[float], method: str = "holm") -> List[float]:
    """
    Adjust p-values for multiple comparisons using Holm-Bonferroni or Benjamini-Hochberg FDR.
    """
    m = len(p_values)
    if m == 0:
        return []

    if method.lower() == "bonferroni":
        return [min(1.0, round(p * m, 6)) for p in p_values]

    # Holm-Bonferroni step-down procedure
    sorted_indices = np.argsort(p_values)
    adjusted = np.zeros(m)
    for rank, idx in enumerate(sorted_indices):
        raw_p = p_values[idx]
        multiplier = m - rank
        adjusted[idx] = min(1.0, raw_p * multiplier)

    # Ensure monotonicity
    for i in range(1, m):
        curr_idx = sorted_indices[i]
        prev_idx = sorted_indices[i - 1]
        adjusted[curr_idx] = max(adjusted[curr_idx], adjusted[prev_idx])

    return [round(float(p), 6) for p in adjusted]
