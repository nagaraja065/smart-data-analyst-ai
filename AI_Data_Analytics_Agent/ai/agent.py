"""
AI Agent — Multi-step Data Analytics Agent with Tool Use.

Routes queries to Claude (if API key) or local pattern engine.
Generates charts, runs analysis, and provides structured responses.

Design Pattern: ReAct Agent (Reasoning + Acting)
SOLID: Dependency Inversion — depends on abstract provider interface.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import re
import time

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

from core.logger import get_logger
from config.settings import settings
from config.constants import COLUMN_KEYWORDS, CHART_COLORS, PLOTLY_TEMPLATE

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """Response from the AI agent."""
    text: str
    chart: Optional[Any] = None
    steps: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class DataAnalyticsAgent:
    """Enterprise AI Agent for data analysis with tool use and multi-step reasoning."""

    def __init__(self, df: pd.DataFrame, api_key: str = ""):
        self.df = df
        self.api_key = api_key
        self.num_cols = df.select_dtypes(include="number").columns.tolist()
        self.cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.date_cols = [c for c in df.columns if any(kw in c.lower() for kw in COLUMN_KEYWORDS.get("date", []))]

    def query(self, question: str) -> AgentResponse:
        """Main entry point — analyze a question about the data."""
        start = time.time()
        if self.api_key:
            result = self._query_claude(question)
        else:
            result = self._query_local(question)
        result.metadata["latency_ms"] = int((time.time() - start) * 1000)
        return result

    # ─── Quick Actions ───────────────────────────────────────────────────

    def quick_analyze(self) -> AgentResponse:
        return self._build_overview()

    def quick_anomalies(self) -> AgentResponse:
        return self._detect_anomalies()

    def quick_insights(self) -> AgentResponse:
        return self._generate_insights()

    def quick_charts(self) -> AgentResponse:
        return self._suggest_charts()

    # ─── Local Pattern Engine ────────────────────────────────────────────

    def _query_local(self, q: str) -> AgentResponse:
        ql = q.lower()
        steps = ["🔍 Analyzing locally..."]

        if any(kw in ql for kw in ["overview", "summary", "describe", "about"]):
            return self._build_overview()
        if any(kw in ql for kw in ["anomal", "outlier", "unusual"]):
            return self._detect_anomalies()
        if any(kw in ql for kw in ["top", "best", "highest", "most", "max"]):
            return self._handle_top(ql, steps)
        if any(kw in ql for kw in ["bottom", "worst", "lowest", "least", "min"]):
            return self._handle_bottom(ql, steps)
        if any(kw in ql for kw in ["total", "sum"]):
            return self._handle_total(ql, steps)
        if any(kw in ql for kw in ["average", "mean", "avg"]):
            return self._handle_average(ql, steps)
        if any(kw in ql for kw in ["trend", "over time", "monthly", "growth"]):
            return self._handle_trend(ql, steps)
        if any(kw in ql for kw in ["distribution", "histogram", "spread"]):
            return self._handle_distribution(ql, steps)
        if any(kw in ql for kw in ["correlation", "correlate", "relationship"]):
            return self._handle_correlation(steps)
        if any(kw in ql for kw in ["compare", "versus", "vs"]):
            return self._handle_compare(ql, steps)
        if any(kw in ql for kw in ["chart", "suggest", "visual", "plot"]):
            return self._suggest_charts()
        if any(kw in ql for kw in ["insight", "recommendation"]):
            return self._generate_insights()

        return self._build_overview()

    def _find_col(self, q: str, col_list: list[str]) -> Optional[str]:
        for col in col_list:
            if col.lower() in q:
                return col
            for word in col.lower().replace("_", " ").split():
                if len(word) > 2 and word in q:
                    return col
        return None

    def _build_overview(self) -> AgentResponse:
        lines = [
            "### 📊 Dataset Overview\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| **Rows** | {len(self.df):,} |",
            f"| **Columns** | {self.df.shape[1]} |",
            f"| **Numeric** | {len(self.num_cols)} |",
            f"| **Categorical** | {len(self.cat_cols)} |",
            f"| **Missing Values** | {int(self.df.isnull().sum().sum())} |",
            f"| **Duplicates** | {int(self.df.duplicated().sum())} |",
            f"| **Memory** | {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB |",
        ]
        for col in self.num_cols[:5]:
            lines.append(f"\n**{col}**: total={self.df[col].sum():,.2f}, mean={self.df[col].mean():,.2f}, "
                         f"min={self.df[col].min():,.2f}, max={self.df[col].max():,.2f}")

        chart = None
        if self.cat_cols and self.num_cols:
            data = self.df.groupby(self.cat_cols[0])[self.num_cols[0]].sum().nlargest(8).reset_index()
            chart = px.bar(data, x=self.cat_cols[0], y=self.num_cols[0],
                           title=f"{self.num_cols[0]} by {self.cat_cols[0]}",
                           template=PLOTLY_TEMPLATE, color=self.cat_cols[0],
                           color_discrete_sequence=CHART_COLORS)

        return AgentResponse("\n".join(lines), chart, ["📋 Built dataset overview"])

    def _detect_anomalies(self) -> AgentResponse:
        findings = ["### 🚨 Anomaly Detection (IQR Method)\n"]
        for col in self.num_cols[:6]:
            q1, q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
            if len(outliers) > 0:
                findings.append(f"- **{col}**: {len(outliers)} outliers (bounds: {lower:,.1f} – {upper:,.1f})")

        chart = None
        if self.num_cols:
            chart = px.box(self.df, y=self.num_cols[:4], title="Outlier Distribution", template=PLOTLY_TEMPLATE)

        if len(findings) == 1:
            findings.append("✅ No significant outliers found.")

        return AgentResponse("\n".join(findings), chart, ["🔎 IQR anomaly scan"])

    def _handle_top(self, q: str, steps: list) -> AgentResponse:
        n = 5
        nums = re.findall(r'\d+', q)
        if nums:
            n = min(int(nums[0]), 20)
        num_col = self._find_col(q, self.num_cols) or (self.num_cols[0] if self.num_cols else None)
        cat_col = self._find_col(q, self.cat_cols) or (self.cat_cols[0] if self.cat_cols else None)

        if cat_col and num_col:
            result = self.df.groupby(cat_col)[num_col].sum().nlargest(n).reset_index()
            fig = px.bar(result, x=cat_col, y=num_col, title=f"Top {n} {cat_col} by {num_col}",
                         template=PLOTLY_TEMPLATE, color=cat_col, color_discrete_sequence=CHART_COLORS)
            return AgentResponse(f"**Top {n} {cat_col} by {num_col}:**\n\n{result.to_markdown(index=False)}", fig, steps)

        return AgentResponse("Please specify columns to analyze.", None, steps)

    def _handle_bottom(self, q: str, steps: list) -> AgentResponse:
        n = 5
        nums = re.findall(r'\d+', q)
        if nums:
            n = min(int(nums[0]), 20)
        num_col = self._find_col(q, self.num_cols) or (self.num_cols[0] if self.num_cols else None)
        cat_col = self._find_col(q, self.cat_cols) or (self.cat_cols[0] if self.cat_cols else None)

        if cat_col and num_col:
            result = self.df.groupby(cat_col)[num_col].sum().nsmallest(n).reset_index()
            fig = px.bar(result, x=cat_col, y=num_col, title=f"Bottom {n} {cat_col} by {num_col}",
                         template=PLOTLY_TEMPLATE, color=cat_col, color_discrete_sequence=CHART_COLORS)
            return AgentResponse(f"**Bottom {n} {cat_col} by {num_col}:**\n\n{result.to_markdown(index=False)}", fig, steps)

        return AgentResponse("Please specify columns to analyze.", None, steps)

    def _handle_total(self, q: str, steps: list) -> AgentResponse:
        col = self._find_col(q, self.num_cols) or (self.num_cols[0] if self.num_cols else None)
        if col:
            return AgentResponse(
                f"### 💰 {col}\n- **Total:** {self.df[col].sum():,.2f}\n"
                f"- **Average:** {self.df[col].mean():,.2f}\n- **Count:** {len(self.df):,}",
                None, steps)
        return AgentResponse("No numeric columns found.", None, steps)

    def _handle_average(self, q: str, steps: list) -> AgentResponse:
        col = self._find_col(q, self.num_cols) or (self.num_cols[0] if self.num_cols else None)
        cat = self._find_col(q, self.cat_cols)
        if col and cat:
            result = self.df.groupby(cat)[col].mean().round(2).reset_index()
            fig = px.bar(result, x=cat, y=col, title=f"Average {col} by {cat}",
                         template=PLOTLY_TEMPLATE, color=cat, color_discrete_sequence=CHART_COLORS)
            return AgentResponse(result.to_markdown(index=False), fig, steps)
        if col:
            return AgentResponse(
                f"**{col}**: Mean={self.df[col].mean():,.2f}, Median={self.df[col].median():,.2f}, "
                f"Std={self.df[col].std():,.2f}", None, steps)
        return AgentResponse("No numeric columns found.", None, steps)

    def _handle_trend(self, q: str, steps: list) -> AgentResponse:
        if not self.date_cols:
            return AgentResponse("No date columns found for trend analysis.", None, steps)
        date_col = self.date_cols[0]
        num_col = self._find_col(q, self.num_cols) or (self.num_cols[0] if self.num_cols else None)
        if not num_col:
            return AgentResponse("No numeric column to trend.", None, steps)
        try:
            dc = self.df.copy()
            dc[date_col] = pd.to_datetime(dc[date_col], errors="coerce")
            monthly = dc.groupby(dc[date_col].dt.to_period("M"))[num_col].sum().reset_index()
            monthly[date_col] = monthly[date_col].astype(str)
            fig = px.line(monthly, x=date_col, y=num_col, title=f"Monthly Trend: {num_col}",
                          markers=True, template=PLOTLY_TEMPLATE)
            fig.update_traces(line=dict(width=3, color=CHART_COLORS[0]))
            return AgentResponse(monthly.to_markdown(index=False), fig, steps)
        except Exception as e:
            return AgentResponse(f"Error: {e}", None, steps)

    def _handle_distribution(self, q: str, steps: list) -> AgentResponse:
        col = self._find_col(q, self.num_cols) or (self.num_cols[0] if self.num_cols else None)
        if col:
            fig = px.histogram(self.df, x=col, nbins=30, title=f"Distribution: {col}",
                               template=PLOTLY_TEMPLATE, color_discrete_sequence=[CHART_COLORS[0]])
            text = (f"**{col}**: Mean={self.df[col].mean():,.2f}, Median={self.df[col].median():,.2f}, "
                    f"Skew={self.df[col].skew():.2f}")
            return AgentResponse(text, fig, steps)
        return AgentResponse("No numeric columns found.", None, steps)

    def _handle_correlation(self, steps: list) -> AgentResponse:
        if len(self.num_cols) < 2:
            return AgentResponse("Need 2+ numeric columns for correlation.", None, steps)
        corr = self.df[self.num_cols].corr().round(3)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                        title="Correlation Heatmap", template=PLOTLY_TEMPLATE)
        return AgentResponse(corr.to_markdown(), fig, steps)

    def _handle_compare(self, q: str, steps: list) -> AgentResponse:
        cat = self._find_col(q, self.cat_cols) or (self.cat_cols[0] if self.cat_cols else None)
        num = self._find_col(q, self.num_cols) or (self.num_cols[0] if self.num_cols else None)
        if cat and num:
            result = self.df.groupby(cat)[num].agg(["sum", "mean", "count"]).round(2).reset_index()
            fig = px.bar(result, x=cat, y="sum", title=f"{num} by {cat}",
                         template=PLOTLY_TEMPLATE, color=cat, color_discrete_sequence=CHART_COLORS)
            return AgentResponse(result.to_markdown(index=False), fig, steps)
        return AgentResponse("Please specify columns to compare.", None, steps)

    def _suggest_charts(self) -> AgentResponse:
        lines = ["### 🎨 Suggested Charts\n"]
        chart = None
        if self.cat_cols and self.num_cols:
            lines.append(f"1. **Bar Chart**: {self.cat_cols[0]} vs {self.num_cols[0]}")
            data = self.df.groupby(self.cat_cols[0])[self.num_cols[0]].sum().nlargest(8).reset_index()
            chart = px.bar(data, x=self.cat_cols[0], y=self.num_cols[0], template=PLOTLY_TEMPLATE,
                           color=self.cat_cols[0], color_discrete_sequence=CHART_COLORS)
        if len(self.num_cols) >= 2:
            lines.append(f"2. **Scatter**: {self.num_cols[0]} vs {self.num_cols[1]}")
        if self.date_cols:
            lines.append(f"3. **Line**: {self.num_cols[0] if self.num_cols else 'N/A'} trend")
        if self.num_cols:
            lines.append(f"4. **Histogram**: Distribution of {self.num_cols[0]}")
        if len(self.num_cols) >= 3:
            lines.append(f"5. **Heatmap**: Correlation matrix")
        return AgentResponse("\n".join(lines), chart, ["🎨 Chart suggestions"])

    def _generate_insights(self) -> AgentResponse:
        lines = ["### 💡 Auto-Generated Insights\n"]
        for col in self.num_cols:
            if any(kw in col.lower() for kw in ["sales", "revenue"]):
                total = self.df[col].sum()
                lines.append(f"- 💰 **{col}**: Total = {total:,.2f}, Avg = {self.df[col].mean():,.2f}")
                if self.cat_cols:
                    top = self.df.groupby(self.cat_cols[0])[col].sum().idxmax()
                    lines.append(f"  - Top {self.cat_cols[0]}: **{top}**")
            if any(kw in col.lower() for kw in ["profit"]):
                neg = (self.df[col] < 0).sum()
                if neg > 0:
                    lines.append(f"- ⚠️ **{neg}** transactions have negative {col}")

        lines.append("\n### 🎯 Recommendations")
        lines.append("- Focus on top-performing categories for maximum ROI")
        lines.append("- Investigate negative profit entries")
        lines.append("- Use time-series forecasting for planning")
        return AgentResponse("\n".join(lines), None, ["💡 Insights generated"])

    # ─── Claude API ──────────────────────────────────────────────────────

    def _query_claude(self, question: str) -> AgentResponse:
        steps = ["🤖 Querying Claude AI..."]
        context = self._build_context()
        prompt = f"""You are an expert data analyst. Analyze the dataset and answer concisely with numbers.

Dataset:
{context}

Question: {question}

Provide a clear answer with bullet points and specific numbers."""

        try:
            resp = requests.post(
                settings.llm.api_url,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": settings.llm.api_version,
                },
                json={
                    "model": settings.llm.model,
                    "max_tokens": settings.llm.max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            data = resp.json()
            if "content" in data and data["content"]:
                text = data["content"][0].get("text", "No response.")
                steps.append("✅ Claude responded")
                chart = self._auto_chart(question)
                return AgentResponse(text, chart, steps)
            else:
                error = data.get("error", {}).get("message", "Unknown error")
                steps.append(f"❌ API error: {error}")
                steps.append("🔄 Falling back to local...")
                return self._query_local(question)
        except Exception as e:
            steps.append(f"❌ {e}")
            return self._query_local(question)

    def _build_context(self) -> str:
        lines = [f"Shape: {self.df.shape[0]}×{self.df.shape[1]}", f"Columns: {', '.join(self.df.columns)}"]
        for col in self.num_cols[:6]:
            lines.append(f"{col}: sum={self.df[col].sum():,.2f}, mean={self.df[col].mean():,.2f}")
        for col in self.cat_cols[:3]:
            lines.append(f"{col}: {self.df[col].value_counts().head(3).to_dict()}")
        return "\n".join(lines)

    def _auto_chart(self, q: str) -> Optional[Any]:
        ql = q.lower()
        try:
            if any(kw in ql for kw in ["top", "best"]) and self.cat_cols and self.num_cols:
                data = self.df.groupby(self.cat_cols[0])[self.num_cols[0]].sum().nlargest(8).reset_index()
                return px.bar(data, x=self.cat_cols[0], y=self.num_cols[0], template=PLOTLY_TEMPLATE,
                              color=self.cat_cols[0], color_discrete_sequence=CHART_COLORS)
            if any(kw in ql for kw in ["trend", "time"]) and self.date_cols and self.num_cols:
                dc = self.df.copy()
                dc[self.date_cols[0]] = pd.to_datetime(dc[self.date_cols[0]], errors="coerce")
                m = dc.groupby(dc[self.date_cols[0]].dt.to_period("M"))[self.num_cols[0]].sum().reset_index()
                m[self.date_cols[0]] = m[self.date_cols[0]].astype(str)
                return px.line(m, x=self.date_cols[0], y=self.num_cols[0], markers=True, template=PLOTLY_TEMPLATE)
        except Exception:
            pass
        return None
