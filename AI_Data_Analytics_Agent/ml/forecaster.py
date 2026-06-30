"""
Time Series Forecaster — Multi-method Forecasting.

Supports Linear Trend, Moving Average, and Random Forest methods.

Design Pattern: Strategy (swap forecasting methods)
SOLID: Single Responsibility — only handles time-series forecasting.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.logger import get_logger
from config.constants import RANDOM_STATE

logger = get_logger(__name__)


@dataclass
class ForecastResult:
    """Time-series forecasting result."""
    method: str
    historical: pd.DataFrame
    forecast: pd.DataFrame
    metrics: dict[str, float]
    model_name: str = ""


class TimeSeriesForecaster:
    """Multi-method time-series forecaster."""

    def forecast(self, df: pd.DataFrame, date_col: str, value_col: str,
                 periods: int = 3, method: str = "linear") -> ForecastResult:
        """
        Forecast future values.

        Args:
            df: DataFrame with time-series data.
            date_col: Date column name.
            value_col: Value column to forecast.
            periods: Number of future periods to predict.
            method: 'linear', 'moving_average', 'random_forest'.
        """
        logger.info(f"Forecasting {value_col} with {method}, {periods} periods ahead")

        # Prepare monthly aggregation
        df_t = df.copy()
        df_t[date_col] = pd.to_datetime(df_t[date_col], errors="coerce")
        df_t = df_t.dropna(subset=[date_col, value_col])

        monthly = df_t.groupby(df_t[date_col].dt.to_period("M"))[value_col].sum().reset_index()
        monthly.columns = ["Period", "Value"]
        monthly["Period_str"] = monthly["Period"].astype(str)
        monthly["Index"] = range(len(monthly))

        if method == "moving_average":
            return self._moving_average(monthly, periods)
        elif method == "random_forest":
            return self._random_forest(monthly, periods)
        else:
            return self._linear_trend(monthly, periods)

    def _linear_trend(self, monthly: pd.DataFrame, periods: int) -> ForecastResult:
        X = monthly[["Index"]].values
        y = monthly["Value"].values

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        metrics = self._calc_metrics(y, y_pred)

        # Forecast
        last_idx = monthly["Index"].max()
        future_idx = np.array([[last_idx + i] for i in range(1, periods + 1)])
        future_vals = model.predict(future_idx)
        std = np.std(y - y_pred)

        forecast_df = pd.DataFrame({
            "Period": [f"Month +{i}" for i in range(1, periods + 1)],
            "Forecast": np.round(future_vals, 2),
            "Lower": np.round(future_vals - 1.96 * std, 2),
            "Upper": np.round(future_vals + 1.96 * std, 2),
        })

        return ForecastResult("Linear Trend", monthly, forecast_df, metrics, "LinearRegression")

    def _moving_average(self, monthly: pd.DataFrame, periods: int) -> ForecastResult:
        window = min(3, len(monthly))
        monthly["MA"] = monthly["Value"].rolling(window=window).mean()
        y = monthly["Value"].values
        y_pred = monthly["MA"].fillna(y[0]).values

        metrics = self._calc_metrics(y, y_pred)

        last_ma = monthly["Value"].tail(window).mean()
        std = monthly["Value"].tail(window).std()

        forecast_df = pd.DataFrame({
            "Period": [f"Month +{i}" for i in range(1, periods + 1)],
            "Forecast": [round(last_ma, 2)] * periods,
            "Lower": [round(last_ma - 1.96 * std, 2)] * periods,
            "Upper": [round(last_ma + 1.96 * std, 2)] * periods,
        })

        return ForecastResult("Moving Average", monthly, forecast_df, metrics, f"MA(window={window})")

    def _random_forest(self, monthly: pd.DataFrame, periods: int) -> ForecastResult:
        X = monthly[["Index"]].values
        y = monthly["Value"].values

        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        model.fit(X, y)
        y_pred = model.predict(X)

        metrics = self._calc_metrics(y, y_pred)

        last_idx = monthly["Index"].max()
        future_idx = np.array([[last_idx + i] for i in range(1, periods + 1)])
        future_vals = model.predict(future_idx)
        std = np.std(y - y_pred)

        forecast_df = pd.DataFrame({
            "Period": [f"Month +{i}" for i in range(1, periods + 1)],
            "Forecast": np.round(future_vals, 2),
            "Lower": np.round(future_vals - 1.96 * std, 2),
            "Upper": np.round(future_vals + 1.96 * std, 2),
        })

        return ForecastResult("Random Forest", monthly, forecast_df, metrics, "RandomForestRegressor")

    def _calc_metrics(self, y: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            "MAE": round(mean_absolute_error(y, y_pred), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y, y_pred)), 2),
            "R²": round(r2_score(y, y_pred), 4),
        }
