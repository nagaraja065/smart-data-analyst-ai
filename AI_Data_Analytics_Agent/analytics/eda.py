"""
EDA Engine — Advanced Exploratory Data Analysis.

Performs univariate, bivariate, and multivariate analysis
with structured reporting.

Design Pattern: Facade (simple interface to complex analysis)
SOLID: Single Responsibility — only performs EDA computations.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import numpy as np

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class UnivariateStats:
    """Statistics for a single numeric column."""
    column: str
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    mode: Any = None
    std: float = 0.0
    variance: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    q1: float = 0.0
    q3: float = 0.0
    iqr: float = 0.0
    outlier_count: int = 0


@dataclass
class CorrelationPair:
    """A pair of correlated columns."""
    col1: str
    col2: str
    correlation: float
    strength: str  # strong, moderate, weak


@dataclass
class EDAReport:
    """Complete EDA report."""
    univariate: list[UnivariateStats] = field(default_factory=list)
    correlation_matrix: Optional[pd.DataFrame] = None
    top_correlations: list[CorrelationPair] = field(default_factory=list)
    categorical_summaries: dict[str, dict] = field(default_factory=dict)
    insights: list[str] = field(default_factory=list)


class EDAEngine:
    """Advanced Exploratory Data Analysis engine."""

    def run_full_eda(self, df: pd.DataFrame) -> EDAReport:
        """Run comprehensive EDA on the DataFrame."""
        logger.info(f"Running EDA on {df.shape[0]}×{df.shape[1]} dataset")
        report = EDAReport()

        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Univariate analysis
        for col in num_cols:
            stats = self._univariate_analysis(df[col])
            report.univariate.append(stats)

        # Correlation analysis
        if len(num_cols) >= 2:
            report.correlation_matrix = df[num_cols].corr().round(4)
            report.top_correlations = self._find_top_correlations(report.correlation_matrix)

        # Categorical summaries
        for col in cat_cols[:10]:
            vc = df[col].value_counts()
            report.categorical_summaries[col] = {
                "unique": int(df[col].nunique()),
                "top_value": str(vc.index[0]) if len(vc) > 0 else "N/A",
                "top_count": int(vc.iloc[0]) if len(vc) > 0 else 0,
                "distribution": vc.head(5).to_dict(),
            }

        # Auto-generate insights
        report.insights = self._generate_insights(df, report, num_cols, cat_cols)

        logger.info(f"EDA complete: {len(report.univariate)} numeric, "
                     f"{len(report.categorical_summaries)} categorical")
        return report

    def _univariate_analysis(self, series: pd.Series) -> UnivariateStats:
        """Compute univariate statistics for a numeric column."""
        s = series.dropna()
        q1 = float(s.quantile(0.25)) if len(s) > 0 else 0
        q3 = float(s.quantile(0.75)) if len(s) > 0 else 0
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = int(((s < lower) | (s > upper)).sum())

        mode_val = s.mode().iloc[0] if not s.mode().empty else None

        return UnivariateStats(
            column=series.name,
            count=len(s),
            mean=round(float(s.mean()), 4) if len(s) > 0 else 0,
            median=round(float(s.median()), 4) if len(s) > 0 else 0,
            mode=mode_val,
            std=round(float(s.std()), 4) if len(s) > 0 else 0,
            variance=round(float(s.var()), 4) if len(s) > 0 else 0,
            skewness=round(float(s.skew()), 4) if len(s) > 0 else 0,
            kurtosis=round(float(s.kurtosis()), 4) if len(s) > 0 else 0,
            min_val=round(float(s.min()), 4) if len(s) > 0 else 0,
            max_val=round(float(s.max()), 4) if len(s) > 0 else 0,
            q1=round(q1, 4),
            q3=round(q3, 4),
            iqr=round(iqr, 4),
            outlier_count=outliers,
        )

    def _find_top_correlations(self, corr_matrix: pd.DataFrame, top_n: int = 10) -> list[CorrelationPair]:
        """Find strongest correlations (excluding self-correlation)."""
        pairs = []
        cols = corr_matrix.columns
        seen = set()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = abs(corr_matrix.iloc[i, j])
                key = (cols[i], cols[j])
                if key not in seen:
                    seen.add(key)
                    strength = "strong" if c >= 0.7 else "moderate" if c >= 0.4 else "weak"
                    pairs.append(CorrelationPair(cols[i], cols[j], round(corr_matrix.iloc[i, j], 4), strength))
        pairs.sort(key=lambda x: abs(x.correlation), reverse=True)
        return pairs[:top_n]

    def _generate_insights(self, df: pd.DataFrame, report: EDAReport,
                            num_cols: list[str], cat_cols: list[str]) -> list[str]:
        """Auto-generate EDA insights."""
        insights = []
        # Skewness insights
        for stat in report.univariate:
            if abs(stat.skewness) > 1:
                direction = "right" if stat.skewness > 0 else "left"
                insights.append(f"'{stat.column}' is highly skewed to the {direction} ({stat.skewness:.2f})")
            if stat.outlier_count > 0:
                pct = stat.outlier_count / stat.count * 100 if stat.count > 0 else 0
                insights.append(f"'{stat.column}' has {stat.outlier_count} outliers ({pct:.1f}%)")

        # Correlation insights
        for pair in report.top_correlations[:3]:
            if pair.strength == "strong":
                insights.append(f"Strong correlation between '{pair.col1}' and '{pair.col2}' ({pair.correlation:.3f})")

        return insights
