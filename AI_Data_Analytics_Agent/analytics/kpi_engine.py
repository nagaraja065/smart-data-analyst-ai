"""
KPI Engine — Automatic Business KPI Detection and Calculation.

Auto-detects business metrics from column names and calculates KPIs.

Design Pattern: Strategy (different KPI strategies for different column types)
SOLID: Open/Closed — add new KPI types without modifying existing logic.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

from core.logger import get_logger
from config.constants import COLUMN_KEYWORDS

logger = get_logger(__name__)


@dataclass
class KPI:
    """A single business KPI."""
    name: str
    value: float
    formatted_value: str
    icon: str
    category: str  # revenue, profit, volume, customer, time
    change_pct: Optional[float] = None
    context: str = ""


@dataclass
class KPIDashboard:
    """Collection of calculated KPIs."""
    kpis: list[KPI] = field(default_factory=list)
    summary: str = ""


class KPIEngine:
    """Auto-detects and calculates business KPIs from DataFrame columns."""

    def _find_col(self, df: pd.DataFrame, keywords: list[str]) -> Optional[str]:
        """Find column matching keywords."""
        for kw in keywords:
            for col in df.columns:
                if kw.lower() in col.lower():
                    return col
        return None

    def _find_numeric_col(self, df: pd.DataFrame, keywords: list[str]) -> Optional[str]:
        """Find a column matching keywords that is also numeric."""
        for kw in keywords:
            for col in df.columns:
                if kw.lower() in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
                    return col
        return None

    def _safe_sum(self, df: pd.DataFrame, col: str) -> float:
        """Safely sum a column, coercing to numeric first if needed."""
        try:
            series = pd.to_numeric(df[col], errors="coerce")
            return float(series.sum())
        except (ValueError, TypeError):
            return 0.0

    def _safe_mean(self, df: pd.DataFrame, col: str) -> float:
        """Safely compute mean, coercing to numeric first if needed."""
        try:
            series = pd.to_numeric(df[col], errors="coerce")
            return float(series.mean())
        except (ValueError, TypeError):
            return 0.0

    def calculate_kpis(self, df: pd.DataFrame) -> KPIDashboard:
        """Detect and calculate all applicable KPIs."""
        logger.info("Calculating business KPIs")
        dashboard = KPIDashboard()

        sales_col = self._find_numeric_col(df, COLUMN_KEYWORDS.get("sales", []))
        profit_col = self._find_numeric_col(df, COLUMN_KEYWORDS.get("profit", []))
        qty_col = self._find_numeric_col(df, COLUMN_KEYWORDS.get("quantity", []))
        customer_col = self._find_col(df, COLUMN_KEYWORDS.get("customer", []))
        date_col = self._find_col(df, COLUMN_KEYWORDS.get("date", []))
        category_col = self._find_col(df, COLUMN_KEYWORDS.get("category", []))
        product_col = self._find_col(df, COLUMN_KEYWORDS.get("product", []))
        region_col = self._find_col(df, COLUMN_KEYWORDS.get("region", []))

        # Revenue KPIs
        try:
            if sales_col:
                total = self._safe_sum(df, sales_col)
                avg = self._safe_mean(df, sales_col)
                dashboard.kpis.append(KPI("Total Revenue", total, f"₹{total:,.0f}", "💰", "revenue"))
                dashboard.kpis.append(KPI("Average Order Value", avg, f"₹{avg:,.0f}", "📊", "revenue"))
                if qty_col:
                    qty_total = self._safe_sum(df, qty_col)
                    rpu = total / qty_total if qty_total > 0 else 0
                    dashboard.kpis.append(KPI("Revenue Per Unit", rpu, f"₹{rpu:,.0f}", "💵", "revenue"))
        except Exception as e:
            logger.warning(f"Revenue KPI error: {e}")

        # Profitability KPIs
        try:
            if profit_col:
                total_profit = self._safe_sum(df, profit_col)
                dashboard.kpis.append(KPI("Total Profit", total_profit, f"₹{total_profit:,.0f}", "📈", "profit"))
                if sales_col:
                    total_sales = self._safe_sum(df, sales_col)
                    if total_sales > 0:
                        margin = total_profit / total_sales * 100
                        dashboard.kpis.append(KPI("Profit Margin", margin, f"{margin:.1f}%", "🎯", "profit"))
                try:
                    neg_profit = int((pd.to_numeric(df[profit_col], errors="coerce") < 0).sum())
                    if neg_profit > 0:
                        dashboard.kpis.append(KPI("Loss Transactions", float(neg_profit), f"{neg_profit}", "⚠️", "profit",
                                                  context=f"{neg_profit/len(df)*100:.1f}% of transactions"))
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Profit KPI error: {e}")

        # Volume KPIs
        try:
            if qty_col:
                total_qty = self._safe_sum(df, qty_col)
                dashboard.kpis.append(KPI("Total Quantity", total_qty, f"{total_qty:,.0f}", "📦", "volume"))
        except Exception as e:
            logger.warning(f"Volume KPI error: {e}")

        # Customer KPIs
        try:
            if customer_col:
                unique_customers = df[customer_col].nunique()
                dashboard.kpis.append(KPI("Unique Customers", float(unique_customers), f"{unique_customers:,}", "👥", "customer"))
                if sales_col:
                    total_sales = self._safe_sum(df, sales_col)
                    avg_spend = total_sales / unique_customers if unique_customers > 0 else 0
                    dashboard.kpis.append(KPI("Avg Spend/Customer", avg_spend, f"₹{avg_spend:,.0f}", "💳", "customer"))
        except Exception as e:
            logger.warning(f"Customer KPI error: {e}")

        # Top performers
        try:
            if product_col and sales_col:
                top = df.groupby(product_col)[sales_col].sum().idxmax()
                dashboard.kpis.append(KPI("Top Product", 0, str(top), "🏆", "performance"))
            if region_col and sales_col:
                top = df.groupby(region_col)[sales_col].sum().idxmax()
                dashboard.kpis.append(KPI("Top Region", 0, str(top), "📍", "performance"))
            if category_col and sales_col:
                top = df.groupby(category_col)[sales_col].sum().idxmax()
                dashboard.kpis.append(KPI("Top Category", 0, str(top), "🗂️", "performance"))
        except Exception as e:
            logger.warning(f"Top performer KPI error: {e}")

        # Time KPIs
        try:
            if date_col and sales_col:
                self._add_time_kpis(df, date_col, sales_col, dashboard)
        except Exception as e:
            logger.warning(f"Time KPI error: {e}")

        dashboard.summary = f"Generated {len(dashboard.kpis)} KPIs from {len(df):,} records"
        logger.info(dashboard.summary)
        return dashboard

    def _add_time_kpis(self, df: pd.DataFrame, date_col: str,
                        value_col: str, dashboard: KPIDashboard) -> None:
        """Calculate time-based KPIs."""
        try:
            df_t = df.copy()
            df_t[date_col] = pd.to_datetime(df_t[date_col], errors="coerce")
            df_t = df_t.dropna(subset=[date_col])
            if len(df_t) == 0:
                return

            monthly = df_t.groupby(df_t[date_col].dt.to_period("M"))[value_col].sum()
            if len(monthly) >= 2:
                last = monthly.iloc[-1]
                prev = monthly.iloc[-2]
                if prev > 0:
                    growth = (last - prev) / prev * 100
                    dashboard.kpis.append(KPI("MoM Growth", growth, f"{growth:+.1f}%", "📊", "time"))

            best_month = str(monthly.idxmax())
            dashboard.kpis.append(KPI("Best Month", float(monthly.max()), best_month, "📅", "time"))
        except (ValueError, TypeError, KeyError):
            pass
