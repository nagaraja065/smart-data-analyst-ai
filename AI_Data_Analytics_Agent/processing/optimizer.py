"""
Data Optimizer — Memory Optimization for Large Datasets.

Reduces DataFrame memory usage through downcasting and categorical conversion.
Critical for handling 1M+ row datasets efficiently.

Design Pattern: Decorator (wraps DataFrame with optimization)
SOLID: Single Responsibility — only optimizes memory, no business logic.
"""

from dataclasses import dataclass

import pandas as pd
import numpy as np

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationReport:
    """Report of memory optimization results."""
    original_memory_mb: float
    optimized_memory_mb: float
    savings_mb: float
    savings_pct: float
    actions: list[str]


class DataOptimizer:
    """Optimizes DataFrame memory usage for large datasets."""

    def optimize(self, df: pd.DataFrame, category_threshold: float = 0.5) -> tuple[pd.DataFrame, OptimizationReport]:
        """
        Optimize DataFrame memory usage.

        Args:
            df: DataFrame to optimize.
            category_threshold: Convert string cols to category if unique ratio < this.

        Returns:
            Tuple of (optimized DataFrame, OptimizationReport).
        """
        original_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        actions = []
        df_opt = df.copy()

        # Downcast integers
        int_cols = df_opt.select_dtypes(include=["int64", "int32"]).columns
        for col in int_cols:
            df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer")
            if df_opt[col].dtype != df[col].dtype:
                actions.append(f"Downcast '{col}': {df[col].dtype} → {df_opt[col].dtype}")

        # Downcast floats
        float_cols = df_opt.select_dtypes(include=["float64"]).columns
        for col in float_cols:
            df_opt[col] = pd.to_numeric(df_opt[col], downcast="float")
            if df_opt[col].dtype != df[col].dtype:
                actions.append(f"Downcast '{col}': {df[col].dtype} → {df_opt[col].dtype}")

        # Convert low-cardinality strings to category
        str_cols = df_opt.select_dtypes(include="object").columns
        for col in str_cols:
            ratio = df_opt[col].nunique() / len(df_opt) if len(df_opt) > 0 else 1
            if ratio < category_threshold:
                df_opt[col] = df_opt[col].astype("category")
                actions.append(f"Categorized '{col}' ({df_opt[col].nunique()} unique values)")

        optimized_mb = df_opt.memory_usage(deep=True).sum() / 1024 / 1024
        savings_mb = original_mb - optimized_mb
        savings_pct = (savings_mb / original_mb * 100) if original_mb > 0 else 0

        report = OptimizationReport(
            original_memory_mb=round(original_mb, 2),
            optimized_memory_mb=round(optimized_mb, 2),
            savings_mb=round(savings_mb, 2),
            savings_pct=round(savings_pct, 1),
            actions=actions,
        )

        logger.info(f"Optimization: {original_mb:.1f}MB → {optimized_mb:.1f}MB ({savings_pct:.1f}% saved)")
        return df_opt, report
