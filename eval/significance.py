"""
Statistical significance testing for ablation experiments.

Provides paired bootstrap resampling and McNemar's test for comparing
two retrieval or generation configurations.

For a dissertation, statistical significance is essential to claim that
one configuration outperforms another. With n=50 queries, you need
bootstrap resampling rather than parametric tests.

References:
  - Efron & Tibshirani, "An Introduction to the Bootstrap" (1993)
  - Koehn, "Statistical Significance Tests for Machine Translation Evaluation" (2004)
"""

from __future__ import annotations

import random
from typing import List, Tuple


def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    n_resamples: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Paired bootstrap resampling test.

    Tests whether system B is significantly better than system A.

    Args:
        scores_a: Per-query metric values for system A
        scores_b: Per-query metric values for system B
        n_resamples: Number of bootstrap resamples
        seed: Random seed for reproducibility

    Returns:
        (mean_diff, p_value, ci_95_half_width)
        - mean_diff: mean(B) - mean(A). Positive means B is better.
        - p_value: fraction of resamples where A >= B (one-sided)
        - ci_95_half_width: half-width of 95% CI on the difference
    """
    assert len(scores_a) == len(scores_b), "Score lists must have same length"
    n = len(scores_a)
    if n == 0:
        return 0.0, 1.0, 0.0

    rng = random.Random(seed)
    observed_diff = sum(scores_b) / n - sum(scores_a) / n

    count_a_wins = 0
    diffs = []

    for _ in range(n_resamples):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        mean_a = sum(scores_a[i] for i in indices) / n
        mean_b = sum(scores_b[i] for i in indices) / n
        diff = mean_b - mean_a
        diffs.append(diff)
        if diff <= 0:
            count_a_wins += 1

    p_value = count_a_wins / n_resamples

    diffs.sort()
    lo = diffs[int(0.025 * n_resamples)]
    hi = diffs[int(0.975 * n_resamples)]
    ci_half = (hi - lo) / 2

    return observed_diff, p_value, ci_half


def mcnemar_test(
    correct_a: List[bool],
    correct_b: List[bool],
) -> Tuple[float, float]:
    """
    McNemar's test for paired binary outcomes (e.g., hit@1).

    Tests whether the disagreement pattern between A and B is significant.

    Args:
        correct_a: Per-query binary outcomes for system A
        correct_b: Per-query binary outcomes for system B

    Returns:
        (chi2_statistic, p_value_approx)
    """
    import math

    assert len(correct_a) == len(correct_b)
    n = len(correct_a)

    # b: A wrong, B right
    # c: A right, B wrong
    b = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)

    if b + c == 0:
        return 0.0, 1.0

    # McNemar's chi-squared (with continuity correction)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # Approximate p-value from chi-squared distribution with df=1
    # Using a simple approximation: P(X > chi2) for chi2(1)
    # For exactness, use scipy.stats.chi2.sf(chi2, 1) in your notebook
    # Here we provide the statistic and a rough p-value
    p_approx = math.exp(-chi2 / 2)  # rough upper bound

    return chi2, p_approx


def format_significance_result(
    name_a: str,
    name_b: str,
    metric_name: str,
    scores_a: List[float],
    scores_b: List[float],
) -> str:
    """Format a significance test result for reporting."""
    n = len(scores_a)
    mean_a = sum(scores_a) / n if n else 0.0
    mean_b = sum(scores_b) / n if n else 0.0

    diff, p, ci = paired_bootstrap_test(scores_a, scores_b)

    sig = ""
    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = "n.s."

    return (
        f"{metric_name}: {name_a}={mean_a:.4f} vs {name_b}={mean_b:.4f} "
        f"(diff={diff:+.4f}, p={p:.4f} {sig}, 95%CI=[{diff-ci:.4f}, {diff+ci:.4f}])"
    )
