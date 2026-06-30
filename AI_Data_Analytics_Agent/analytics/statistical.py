"""
Statistical Analyzer — Hypothesis Testing and Statistical Analysis.

Provides normality tests, correlation tests, group comparisons, and outlier detection.

Design Pattern: Template Method (consistent result format across tests)
SOLID: Single Responsibility — only performs statistical tests.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    interpretation: str
    alpha: float = 0.05


@dataclass
class OutlierResult:
    """Result of outlier detection."""
    method: str
    column: str
    outlier_count: int
    outlier_pct: float
    lower_bound: float
    upper_bound: float
    outlier_indices: list[int]


class StatisticalAnalyzer:
    """Statistical hypothesis testing and analysis."""

    def test_normality(self, series: pd.Series, alpha: float = 0.05) -> TestResult:
        """Shapiro-Wilk test for normality."""
        s = series.dropna()
        sample = s.sample(min(len(s), 5000), random_state=42) if len(s) > 5000 else s

        stat, p = scipy_stats.shapiro(sample)
        is_sig = p < alpha

        interp = (f"'{series.name}' does NOT follow a normal distribution (p={p:.4f})"
                  if is_sig else
                  f"'{series.name}' appears normally distributed (p={p:.4f})")

        return TestResult("Shapiro-Wilk", round(stat, 4), round(p, 4), is_sig, interp, alpha)

    def test_correlation(self, s1: pd.Series, s2: pd.Series, alpha: float = 0.05) -> dict[str, TestResult]:
        """Pearson and Spearman correlation tests."""
        mask = s1.notna() & s2.notna()
        x, y = s1[mask], s2[mask]

        results = {}
        # Pearson
        stat, p = scipy_stats.pearsonr(x, y)
        results["pearson"] = TestResult(
            "Pearson Correlation", round(stat, 4), round(p, 4), p < alpha,
            f"Pearson r={stat:.3f}: {'significant' if p < alpha else 'not significant'} "
            f"{'strong' if abs(stat) >= 0.7 else 'moderate' if abs(stat) >= 0.4 else 'weak'} correlation",
            alpha
        )
        # Spearman
        stat, p = scipy_stats.spearmanr(x, y)
        results["spearman"] = TestResult(
            "Spearman Correlation", round(stat, 4), round(p, 4), p < alpha,
            f"Spearman ρ={stat:.3f}: {'significant' if p < alpha else 'not significant'} "
            f"{'monotonic' if abs(stat) >= 0.7 else 'moderate' if abs(stat) >= 0.4 else 'weak'} relationship",
            alpha
        )
        return results

    def test_group_difference(self, df: pd.DataFrame, group_col: str,
                               value_col: str, alpha: float = 0.05) -> TestResult:
        """t-test (2 groups) or ANOVA (3+ groups)."""
        groups = [group[value_col].dropna().values for _, group in df.groupby(group_col)]
        groups = [g for g in groups if len(g) >= 2]

        if len(groups) < 2:
            return TestResult("N/A", 0.0, 1.0, False, "Need at least 2 groups with data", alpha)

        if len(groups) == 2:
            stat, p = scipy_stats.ttest_ind(groups[0], groups[1])
            test_name = "Independent t-test"
            interp = (f"Significant difference between groups (p={p:.4f})"
                      if p < alpha else
                      f"No significant difference between groups (p={p:.4f})")
        else:
            stat, p = scipy_stats.f_oneway(*groups)
            test_name = f"One-way ANOVA ({len(groups)} groups)"
            interp = (f"Significant difference among {len(groups)} groups (p={p:.4f})"
                      if p < alpha else
                      f"No significant difference among {len(groups)} groups (p={p:.4f})")

        return TestResult(test_name, round(stat, 4), round(p, 4), p < alpha, interp, alpha)

    def detect_outliers(self, series: pd.Series, method: str = "iqr") -> OutlierResult:
        """Detect outliers using IQR or Z-score method."""
        s = series.dropna()

        if method == "zscore":
            z = np.abs(scipy_stats.zscore(s))
            mask = z > 3
            lower = float(s.mean() - 3 * s.std())
            upper = float(s.mean() + 3 * s.std())
        else:  # iqr
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = float(q1 - 1.5 * iqr)
            upper = float(q3 + 1.5 * iqr)
            mask = (s < lower) | (s > upper)

        outlier_idx = s[mask].index.tolist()

        return OutlierResult(
            method=method,
            column=str(series.name),
            outlier_count=int(mask.sum()),
            outlier_pct=round(mask.sum() / len(s) * 100, 2) if len(s) > 0 else 0,
            lower_bound=round(lower, 2),
            upper_bound=round(upper, 2),
            outlier_indices=outlier_idx[:100],
        )
