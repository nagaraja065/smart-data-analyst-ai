"""
Anomaly Detector — Multiple Anomaly Detection Methods.

Supports IQR, Z-Score, Isolation Forest, and Local Outlier Factor.

Design Pattern: Strategy (swap detection methods)
SOLID: Single Responsibility — only detects anomalies.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    method: str
    anomaly_count: int
    anomaly_pct: float
    anomaly_mask: Optional[pd.Series] = None
    details: Optional[pd.DataFrame] = None
    column_results: list[dict] = field(default_factory=list)


class AnomalyDetector:
    """Multi-method anomaly detection."""

    def detect(self, df: pd.DataFrame, features: list[str],
               method: str = "iqr", contamination: float = 0.05) -> AnomalyResult:
        """
        Detect anomalies using specified method.

        Args:
            df: DataFrame.
            features: Numeric columns to analyze.
            method: 'iqr', 'zscore', 'isolation_forest', 'lof'.
            contamination: Expected proportion of anomalies (for IF/LOF).
        """
        logger.info(f"Detecting anomalies with {method} on {len(features)} features")

        if method == "iqr":
            return self._detect_iqr(df, features)
        elif method == "zscore":
            return self._detect_zscore(df, features)
        elif method == "isolation_forest":
            return self._detect_isolation_forest(df, features, contamination)
        elif method == "lof":
            return self._detect_lof(df, features, contamination)
        else:
            return self._detect_iqr(df, features)

    def _detect_iqr(self, df: pd.DataFrame, features: list[str]) -> AnomalyResult:
        mask = pd.Series(False, index=df.index)
        col_results = []
        for col in features:
            s = df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            col_mask = (df[col] < lower) | (df[col] > upper)
            mask = mask | col_mask.fillna(False)
            col_results.append({
                "column": col, "outliers": int(col_mask.sum()),
                "lower": round(lower, 2), "upper": round(upper, 2)
            })
        count = int(mask.sum())
        return AnomalyResult("IQR", count, round(count / len(df) * 100, 2) if len(df) > 0 else 0,
                             mask, df[mask].head(50), col_results)

    def _detect_zscore(self, df: pd.DataFrame, features: list[str]) -> AnomalyResult:
        from scipy import stats
        mask = pd.Series(False, index=df.index)
        col_results = []
        for col in features:
            s = df[col].dropna()
            z = np.abs(stats.zscore(s))
            col_mask = pd.Series(False, index=df.index)
            col_mask.loc[s.index] = z > 3
            mask = mask | col_mask
            col_results.append({"column": col, "outliers": int(col_mask.sum()), "threshold": 3.0})
        count = int(mask.sum())
        return AnomalyResult("Z-Score", count, round(count / len(df) * 100, 2) if len(df) > 0 else 0,
                             mask, df[mask].head(50), col_results)

    def _detect_isolation_forest(self, df: pd.DataFrame, features: list[str],
                                  contamination: float) -> AnomalyResult:
        data = df[features].dropna()
        model = IsolationForest(contamination=contamination, random_state=42)
        preds = model.fit_predict(data)
        mask = pd.Series(False, index=df.index)
        mask.loc[data.index] = preds == -1
        count = int(mask.sum())
        return AnomalyResult("Isolation Forest", count, round(count / len(df) * 100, 2) if len(df) > 0 else 0,
                             mask, df[mask].head(50))

    def _detect_lof(self, df: pd.DataFrame, features: list[str],
                     contamination: float) -> AnomalyResult:
        data = df[features].dropna()
        model = LocalOutlierFactor(contamination=contamination)
        preds = model.fit_predict(data)
        mask = pd.Series(False, index=df.index)
        mask.loc[data.index] = preds == -1
        count = int(mask.sum())
        return AnomalyResult("Local Outlier Factor", count, round(count / len(df) * 100, 2) if len(df) > 0 else 0,
                             mask, df[mask].head(50))
