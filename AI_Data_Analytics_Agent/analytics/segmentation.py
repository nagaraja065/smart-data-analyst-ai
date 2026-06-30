"""
Segmentation Engine — Customer and Category Segmentation.

Implements RFM segmentation and category performance ranking.

Design Pattern: Strategy (different segmentation strategies)
SOLID: Single Responsibility — only performs segmentation analysis.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

from core.logger import get_logger

logger = get_logger(__name__)

RFM_SEGMENTS = {
    (5, 5, 5): "Champions", (5, 5, 4): "Champions", (5, 4, 5): "Champions",
    (4, 5, 5): "Loyal", (5, 4, 4): "Loyal", (4, 4, 5): "Loyal", (4, 5, 4): "Loyal",
    (5, 3, 3): "Potential", (4, 3, 3): "Potential", (3, 3, 3): "Potential",
    (5, 1, 1): "New Customers", (5, 1, 2): "New Customers", (5, 2, 1): "New Customers",
    (2, 2, 2): "At Risk", (2, 3, 3): "At Risk", (3, 2, 2): "At Risk",
    (1, 1, 1): "Lost", (1, 2, 1): "Lost", (1, 1, 2): "Lost", (2, 1, 1): "Lost",
}


@dataclass
class RFMResult:
    """RFM segmentation result."""
    rfm_table: pd.DataFrame
    segment_counts: dict[str, int]
    segment_descriptions: dict[str, str]
    total_customers: int = 0


@dataclass
class CategoryReport:
    """Category performance report."""
    rankings: pd.DataFrame
    top_performer: str = ""
    bottom_performer: str = ""
    insights: list[str] = field(default_factory=list)


class SegmentationEngine:
    """Customer and product segmentation analysis."""

    def rfm_segmentation(self, df: pd.DataFrame, customer_col: str,
                          date_col: str, value_col: str) -> RFMResult:
        """
        Perform RFM (Recency, Frequency, Monetary) segmentation.

        Args:
            df: DataFrame with transaction data.
            customer_col: Column with customer identifiers.
            date_col: Column with transaction dates.
            value_col: Column with transaction values.

        Returns:
            RFMResult with segments and distributions.
        """
        logger.info(f"Running RFM segmentation on {df[customer_col].nunique()} customers")

        df_rfm = df.copy()
        df_rfm[date_col] = pd.to_datetime(df_rfm[date_col], errors="coerce")
        df_rfm = df_rfm.dropna(subset=[date_col, customer_col, value_col])

        max_date = df_rfm[date_col].max()

        rfm = df_rfm.groupby(customer_col).agg({
            date_col: lambda x: (max_date - x.max()).days,  # Recency
            customer_col: "count",                           # Frequency (using customer_col as proxy)
            value_col: "sum",                                # Monetary
        })
        # Rename properly — the customer_col count is Frequency
        rfm.columns = ["Recency", "Frequency", "Monetary"]

        # Score 1-5 using quantiles
        for metric in ["Recency", "Frequency", "Monetary"]:
            try:
                if metric == "Recency":
                    rfm[f"{metric}_Score"] = pd.qcut(rfm[metric], 5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
                else:
                    rfm[f"{metric}_Score"] = pd.qcut(rfm[metric], 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
            except (ValueError, TypeError):
                rfm[f"{metric}_Score"] = 3  # Default to middle if can't quantile

        # Assign segments
        rfm["Segment"] = rfm.apply(
            lambda row: self._get_segment(row["Recency_Score"], row["Frequency_Score"], row["Monetary_Score"]),
            axis=1
        )

        segment_counts = rfm["Segment"].value_counts().to_dict()

        descriptions = {
            "Champions": "Best customers — high recency, frequency, and monetary value",
            "Loyal": "Frequent buyers with good spending — nurture these relationships",
            "Potential": "Recent customers with growth potential — encourage more purchases",
            "New Customers": "Recently acquired — create a strong first impression",
            "At Risk": "Declining engagement — re-engagement campaigns needed",
            "Lost": "Haven't purchased in a long time — win-back campaigns",
            "Other": "Moderate engagement — targeted promotions recommended",
        }

        return RFMResult(
            rfm_table=rfm.reset_index(),
            segment_counts=segment_counts,
            segment_descriptions=descriptions,
            total_customers=len(rfm),
        )

    def _get_segment(self, r: int, f: int, m: int) -> str:
        """Map RFM scores to segment name."""
        return RFM_SEGMENTS.get((r, f, m), "Other")

    def category_performance(self, df: pd.DataFrame, category_col: str,
                              value_col: str) -> CategoryReport:
        """
        Rank categories by performance.

        Args:
            df: DataFrame.
            category_col: Column with categories.
            value_col: Numeric column for ranking.

        Returns:
            CategoryReport with rankings and insights.
        """
        logger.info(f"Analyzing category performance: {category_col} by {value_col}")

        rankings = df.groupby(category_col).agg(
            Total=(value_col, "sum"),
            Average=(value_col, "mean"),
            Count=(value_col, "count"),
            Std=(value_col, "std"),
        ).round(2)

        rankings["Share_%"] = (rankings["Total"] / rankings["Total"].sum() * 100).round(1)
        rankings = rankings.sort_values("Total", ascending=False).reset_index()
        rankings.insert(0, "Rank", range(1, len(rankings) + 1))

        top = rankings.iloc[0][category_col] if len(rankings) > 0 else ""
        bottom = rankings.iloc[-1][category_col] if len(rankings) > 0 else ""

        insights = []
        if len(rankings) > 0:
            top_share = rankings.iloc[0]["Share_%"]
            insights.append(f"'{top}' leads with {top_share}% market share")
        if len(rankings) > 1:
            ratio = rankings.iloc[0]["Total"] / rankings.iloc[-1]["Total"] if rankings.iloc[-1]["Total"] > 0 else 0
            insights.append(f"Top-to-bottom ratio: {ratio:.1f}x")

        return CategoryReport(rankings=rankings, top_performer=str(top),
                               bottom_performer=str(bottom), insights=insights)
