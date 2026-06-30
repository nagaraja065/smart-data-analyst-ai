"""
Data Cleaner — Configurable Cleaning Pipeline.

Applies cleaning transformations in a configurable pipeline:
remove duplicates → fill nulls → parse dates → strip whitespace → drop empty cols.

Design Pattern: Pipeline (each step is independent and composable)
SOLID: Open/Closed — add new cleaning steps without modifying existing ones.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CleaningConfig:
    """Configuration for which cleaning steps to apply."""
    remove_duplicates: bool = True
    fill_numeric_nulls: bool = True
    fill_categorical_nulls: bool = True
    parse_dates: bool = True
    strip_whitespace: bool = True
    drop_empty_columns: bool = True
    numeric_fill_strategy: str = "median"    # median, mean, zero
    categorical_fill_strategy: str = "mode"  # mode, unknown


@dataclass
class CleaningReport:
    """Report of all cleaning actions taken."""
    actions: list[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    columns_before: int = 0
    columns_after: int = 0
    nulls_filled: int = 0
    duplicates_removed: int = 0
    columns_dropped: list[str] = field(default_factory=list)


class DataCleaner:
    """Configurable data cleaning pipeline."""

    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
        """
        Apply cleaning pipeline to DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (cleaned DataFrame, CleaningReport).
        """
        report = CleaningReport(rows_before=len(df), columns_before=df.shape[1])
        df_clean = df.copy()
        logger.info(f"Starting cleaning pipeline: {len(df)} rows × {df.shape[1]} cols")

        if self.config.remove_duplicates:
            df_clean = self._remove_duplicates(df_clean, report)
        if self.config.fill_numeric_nulls:
            df_clean = self._fill_numeric(df_clean, report)
        if self.config.fill_categorical_nulls:
            df_clean = self._fill_categorical(df_clean, report)
        if self.config.parse_dates:
            df_clean = self._parse_dates(df_clean, report)
        if self.config.strip_whitespace:
            df_clean = self._strip_whitespace(df_clean, report)
        if self.config.drop_empty_columns:
            df_clean = self._drop_empty_columns(df_clean, report)

        report.rows_after = len(df_clean)
        report.columns_after = df_clean.shape[1]
        logger.info(f"Cleaning complete: {len(report.actions)} actions applied")
        return df_clean, report

    def _remove_duplicates(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        dupes = int(df.duplicated().sum())
        if dupes > 0:
            df = df.drop_duplicates()
            report.duplicates_removed = dupes
            report.actions.append(f"Removed {dupes} duplicate rows")
            logger.info(f"Removed {dupes} duplicates")
        return df

    def _fill_numeric(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        num_cols = df.select_dtypes(include="number").columns
        strategy = self.config.numeric_fill_strategy
        for col in num_cols:
            missing = int(df[col].isnull().sum())
            if missing > 0:
                if strategy == "median":
                    fill_val = df[col].median()
                elif strategy == "mean":
                    fill_val = df[col].mean()
                else:
                    fill_val = 0
                df[col] = df[col].fillna(fill_val)
                report.nulls_filled += missing
                report.actions.append(f"Filled {missing} nulls in '{col}' with {strategy} ({fill_val:.2f})")
        return df

    def _fill_categorical(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        strategy = self.config.categorical_fill_strategy
        for col in cat_cols:
            missing = int(df[col].isnull().sum())
            if missing > 0:
                if strategy == "mode" and not df[col].mode().empty:
                    fill_val = df[col].mode().iloc[0]
                else:
                    fill_val = "Unknown"
                df[col] = df[col].fillna(fill_val)
                report.nulls_filled += missing
                report.actions.append(f"Filled {missing} nulls in '{col}' with '{fill_val}'")
        return df

    def _parse_dates(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        for col in df.columns:
            if any(kw in col.lower() for kw in ["date", "time", "timestamp"]):
                if df[col].dtype == "object":
                    try:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                        report.actions.append(f"Parsed '{col}' as datetime")
                    except (ValueError, TypeError):
                        pass
        return df

    def _strip_whitespace(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        str_cols = df.select_dtypes(include="object").columns
        if len(str_cols) > 0:
            for col in str_cols:
                df[col] = df[col].str.strip()
            report.actions.append(f"Stripped whitespace from {len(str_cols)} text columns")
        return df

    def _drop_empty_columns(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        empty_cols = [col for col in df.columns if df[col].isnull().all()]
        if empty_cols:
            df = df.drop(columns=empty_cols)
            report.columns_dropped = empty_cols
            report.actions.append(f"Dropped {len(empty_cols)} empty columns: {', '.join(empty_cols)}")
        return df
