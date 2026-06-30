"""
Data Validator — Schema Validation and Quality Checks.

Validates DataFrames against quality rules: missing values, duplicates,
type consistency, cardinality, and constant columns.

Design Pattern: Chain of Responsibility (each check is independent)
SOLID: Single Responsibility — only validates, never modifies data.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np

from core.logger import get_logger
from config.constants import MAX_MISSING_PCT_WARNING, MAX_MISSING_PCT_CRITICAL, MAX_CARDINALITY_RATIO

logger = get_logger(__name__)


@dataclass
class ColumnReport:
    """Validation report for a single column."""
    name: str
    dtype: str
    missing_count: int = 0
    missing_pct: float = 0.0
    unique_count: int = 0
    cardinality_ratio: float = 0.0
    is_constant: bool = False
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Complete dataset validation report."""
    is_valid: bool = True
    total_rows: int = 0
    total_columns: int = 0
    duplicate_rows: int = 0
    total_missing: int = 0
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    column_reports: list[ColumnReport] = field(default_factory=list)

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)


class DataValidator:
    """Validates DataFrames for quality issues without modifying data."""

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Run all validation checks on a DataFrame.

        Args:
            df: DataFrame to validate.

        Returns:
            ValidationReport with issues, warnings, and per-column details.
        """
        logger.info(f"Validating dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        report = ValidationReport(total_rows=len(df), total_columns=df.shape[1])

        self._check_empty(df, report)
        if not report.is_valid:
            return report

        self._check_duplicates(df, report)
        self._check_missing(df, report)
        self._check_columns(df, report)

        report.is_valid = len(report.issues) == 0
        logger.info(f"Validation complete: {report.issue_count} issues, {report.warning_count} warnings")
        return report

    def _check_empty(self, df: pd.DataFrame, report: ValidationReport) -> None:
        if len(df) == 0:
            report.issues.append("Dataset is empty (0 rows)")
            report.is_valid = False
        if df.shape[1] == 0:
            report.issues.append("Dataset has no columns")
            report.is_valid = False

    def _check_duplicates(self, df: pd.DataFrame, report: ValidationReport) -> None:
        dupes = int(df.duplicated().sum())
        report.duplicate_rows = dupes
        if dupes > 0:
            pct = dupes / len(df) * 100
            report.warnings.append(f"{dupes} duplicate rows ({pct:.1f}%)")

    def _check_missing(self, df: pd.DataFrame, report: ValidationReport) -> None:
        total = int(df.isnull().sum().sum())
        report.total_missing = total
        if total > 0:
            pct = total / (len(df) * df.shape[1]) * 100
            if pct > MAX_MISSING_PCT_CRITICAL:
                report.issues.append(f"{total} missing values ({pct:.1f}%) — critical")
            elif pct > MAX_MISSING_PCT_WARNING:
                report.warnings.append(f"{total} missing values ({pct:.1f}%)")

    def _check_columns(self, df: pd.DataFrame, report: ValidationReport) -> None:
        for col in df.columns:
            col_report = ColumnReport(
                name=col,
                dtype=str(df[col].dtype),
                missing_count=int(df[col].isnull().sum()),
                missing_pct=round(df[col].isnull().sum() / len(df) * 100, 2) if len(df) > 0 else 0,
                unique_count=int(df[col].nunique()),
                cardinality_ratio=round(df[col].nunique() / len(df), 4) if len(df) > 0 else 0,
                is_constant=df[col].nunique() <= 1,
            )

            if col_report.is_constant:
                col_report.warnings.append("Constant column (0 variance)")
            if col_report.missing_pct > MAX_MISSING_PCT_CRITICAL:
                col_report.issues.append(f"{col_report.missing_pct}% missing — consider dropping")
            elif col_report.missing_pct > MAX_MISSING_PCT_WARNING:
                col_report.warnings.append(f"{col_report.missing_pct}% missing")
            if col_report.cardinality_ratio > MAX_CARDINALITY_RATIO and df[col].dtype == "object":
                col_report.warnings.append("High cardinality — possible ID column")

            report.column_reports.append(col_report)
