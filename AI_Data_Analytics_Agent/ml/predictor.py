"""
ML Predictor — Regression and Classification Models.

Provides a unified interface for training, evaluating, and predicting
with multiple sklearn models.

Design Pattern: Strategy (swap models via string name)
SOLID: Open/Closed — add new models without modifying existing code.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from core.logger import get_logger
from config.constants import RANDOM_STATE, TEST_SIZE_DEFAULT

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Result from model training."""
    model_name: str
    model_type: str  # regression or classification
    metrics: dict[str, float]
    feature_importance: Optional[pd.DataFrame] = None
    training_time_ms: int = 0
    predictions: Optional[np.ndarray] = None
    actuals: Optional[np.ndarray] = None
    confusion_mat: Optional[np.ndarray] = None


REGRESSION_MODELS = {
    "Linear Regression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "SVM": SVC,
    "Gradient Boosting": GradientBoostingClassifier,
    "KNN": KNeighborsClassifier,
}


class MLPredictor:
    """Unified ML predictor for regression and classification."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}

    def preprocess(self, df: pd.DataFrame, target_col: str,
                   feature_cols: list[str]) -> tuple:
        """Preprocess data: encode, scale, split."""
        df_proc = df[feature_cols + [target_col]].dropna().copy()

        # Encode categorical features
        for col in feature_cols:
            if df_proc[col].dtype == "object":
                le = LabelEncoder()
                df_proc[col] = le.fit_transform(df_proc[col].astype(str))
                self.label_encoders[col] = le

        X = df_proc[feature_cols].values
        y = df_proc[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE_DEFAULT, random_state=RANDOM_STATE
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_regression(self, df: pd.DataFrame, target_col: str,
                         feature_cols: list[str],
                         model_name: str = "Random Forest") -> TrainingResult:
        """Train a regression model."""
        logger.info(f"Training regression: {model_name}")
        start = time.time()

        X_train, X_test, y_train, y_test = self.preprocess(df, target_col, feature_cols)

        model_class = REGRESSION_MODELS.get(model_name, RandomForestRegressor)
        if model_name in ("Random Forest", "Gradient Boosting"):
            self.model = model_class(n_estimators=100, random_state=RANDOM_STATE)
        else:
            self.model = model_class()

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        elapsed = int((time.time() - start) * 1000)

        metrics = {
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "R²": round(r2_score(y_test, y_pred), 4),
        }

        fi = self._get_feature_importance(feature_cols)
        logger.info(f"Regression complete: R²={metrics['R²']}")

        return TrainingResult(
            model_name=model_name, model_type="regression",
            metrics=metrics, feature_importance=fi,
            training_time_ms=elapsed, predictions=y_pred, actuals=y_test
        )

    def train_classification(self, df: pd.DataFrame, target_col: str,
                              feature_cols: list[str],
                              model_name: str = "Random Forest") -> TrainingResult:
        """Train a classification model."""
        logger.info(f"Training classification: {model_name}")
        start = time.time()

        # Encode target if categorical
        df_proc = df.copy()
        target_le = None
        if df_proc[target_col].dtype == "object":
            target_le = LabelEncoder()
            df_proc[target_col] = target_le.fit_transform(df_proc[target_col].astype(str))

        X_train, X_test, y_train, y_test = self.preprocess(df_proc, target_col, feature_cols)

        model_class = CLASSIFICATION_MODELS.get(model_name, RandomForestClassifier)
        if model_name in ("Random Forest", "Gradient Boosting"):
            self.model = model_class(n_estimators=100, random_state=RANDOM_STATE)
        elif model_name == "Logistic Regression":
            self.model = model_class(max_iter=1000, random_state=RANDOM_STATE)
        elif model_name == "KNN":
            self.model = model_class(n_neighbors=5)
        else:
            self.model = model_class(random_state=RANDOM_STATE)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        elapsed = int((time.time() - start) * 1000)

        avg = "weighted" if len(np.unique(y_test)) > 2 else "binary"
        metrics = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "F1": round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
        }

        fi = self._get_feature_importance(feature_cols)
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Classification complete: Accuracy={metrics['Accuracy']}")

        return TrainingResult(
            model_name=model_name, model_type="classification",
            metrics=metrics, feature_importance=fi,
            training_time_ms=elapsed, predictions=y_pred, actuals=y_test,
            confusion_mat=cm
        )

    def _get_feature_importance(self, feature_cols: list[str]) -> Optional[pd.DataFrame]:
        """Extract feature importance from tree-based models."""
        if hasattr(self.model, "feature_importances_"):
            fi = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": self.model.feature_importances_
            }).sort_values("Importance", ascending=False)
            return fi
        elif hasattr(self.model, "coef_"):
            coefs = self.model.coef_
            if coefs.ndim > 1:
                coefs = coefs[0]
            fi = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": np.abs(coefs)
            }).sort_values("Importance", ascending=False)
            return fi
        return None
