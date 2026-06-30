"""
Data Profiler — Automated Dataset Profiling with Quality Scoring.

Generates comprehensive profiles: per-column stats, type detection,
role detection, and an overall data quality score (0-100).

Design Pattern: Template Method (fixed profiling steps, customizable scoring)
SOLID: Single Responsibility — only profiles, never modifies data.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import numpy as np

from core.logger import get_logger
from config.constants import COLUMN_KEYWORDS

logger = get_logger(__name__)


@dataclass
class ColumnProfile:
    """Profile for a single column."""
    name: str
    dtype: str
    role: str = "unknown"          # sales, profit, date, category, etc.
    missing_count: int = 0
    missing_pct: float = 0.0
    unique_count: int = 0
    top_values: dict = field(default_factory=dict)
    # Numeric stats
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    skew: Optional[float] = None
    kurtosis: Optional[float] = None


@dataclass
class DataProfile:
    """Complete dataset profile."""
    row_count: int = 0
    column_count: int = 0
    memory_mb: float = 0.0
    duplicate_rows: int = 0
    total_missing: int = 0
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)
    column_profiles: list[ColumnProfile] = field(default_factory=list)
    quality_score: float = 0.0
    detected_roles: dict[str, str] = field(default_factory=dict)


class DataProfiler:
    """Auto-profiles datasets with quality scoring and role detection."""

    def profile(self, df: pd.DataFrame) -> DataProfile:
        """
        Generate a comprehensive profile of the DataFrame.

        Args:
            df: DataFrame to profile.

        Returns:
            DataProfile with per-column details and quality score.
        """
        logger.info(f"Profiling dataset: {df.shape[0]} × {df.shape[1]}")

        profile = DataProfile(
            row_count=len(df),
            column_count=df.shape[1],
            memory_mb=round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            duplicate_rows=int(df.duplicated().sum()),
            total_missing=int(df.isnull().sum().sum()),
            numeric_columns=df.select_dtypes(include="number").columns.tolist(),
            categorical_columns=df.select_dtypes(include=["object", "category"]).columns.tolist(),
        )

        # Detect date columns
        for col in df.columns:
            if any(kw in col.lower() for kw in COLUMN_KEYWORDS.get("date", [])):
                profile.date_columns.append(col)

        # Profile each column
        for col in df.columns:
            cp = self._profile_column(df, col)
            profile.column_profiles.append(cp)
            if cp.role != "unknown":
                profile.detected_roles[col] = cp.role

        # Calculate quality score
        profile.quality_score = self._calculate_quality_score(df, profile)

        logger.info(f"Profile complete: quality={profile.quality_score:.1f}/100, "
                     f"roles detected={len(profile.detected_roles)}")
        return profile

    def _profile_column(self, df: pd.DataFrame, col: str) -> ColumnProfile:
        """Profile a single column."""
        series = df[col]
        cp = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            missing_count=int(series.isnull().sum()),
            missing_pct=round(series.isnull().sum() / len(df) * 100, 2) if len(df) > 0 else 0,
            unique_count=int(series.nunique()),
            role=self._detect_role(col),
        )

        # Top values for non-numeric
        if series.dtype == "object" or str(series.dtype) == "category":
            cp.top_values = series.value_counts().head(5).to_dict()

        # Numeric stats
        if pd.api.types.is_numeric_dtype(series):
            cp.mean = round(float(series.mean()), 4) if not series.empty else None
            cp.median = round(float(series.median()), 4) if not series.empty else None
            cp.std = round(float(series.std()), 4) if not series.empty else None
            cp.min_val = round(float(series.min()), 4) if not series.empty else None
            cp.max_val = round(float(series.max()), 4) if not series.empty else None
            try:
                cp.skew = round(float(series.skew()), 4)
                cp.kurtosis = round(float(series.kurtosis()), 4)
            except (ValueError, TypeError):
                pass

        return cp

    def _detect_role(self, col_name: str) -> str:
        """Detect the business role of a column from its name."""
        col_lower = col_name.lower()
        for role, keywords in COLUMN_KEYWORDS.items():
            for kw in keywords:
                if kw in col_lower:
                    return role
        return "unknown"

    def _calculate_quality_score(self, df: pd.DataFrame, profile: DataProfile) -> float:
        """
        Calculate data quality score (0-100).

        Deductions:
        - Missing values: up to -30 points
        - Duplicates: up to -15 points
        - Constant columns: up to -10 points
        - Type issues: up to -10 points
        """
        score = 100.0

        # Missing penalty (up to 30)
        if profile.row_count > 0 and profile.column_count > 0:
            total_cells = profile.row_count * profile.column_count
            missing_pct = profile.total_missing / total_cells * 100
            score -= min(30, missing_pct * 1.5)

        # Duplicate penalty (up to 15)
        if profile.row_count > 0:
            dupe_pct = profile.duplicate_rows / profile.row_count * 100
            score -= min(15, dupe_pct * 0.5)

        # Constant columns penalty (up to 10)
        constant_cols = sum(1 for cp in profile.column_profiles if cp.unique_count <= 1)
        if profile.column_count > 0:
            score -= min(10, constant_cols / profile.column_count * 30)

        # Mixed type penalty (up to 10)
        object_cols_with_numbers = 0
        for cp in profile.column_profiles:
            if cp.dtype == "object" and cp.unique_count > 0:
                sample = df[cp.name].dropna().head(100)
                numeric_count = pd.to_numeric(sample, errors="coerce").notna().sum()
                if numeric_count > len(sample) * 0.5:
                    object_cols_with_numbers += 1
        if profile.column_count > 0:
            score -= min(10, object_cols_with_numbers / profile.column_count * 30)

        return round(max(0, score), 1)
