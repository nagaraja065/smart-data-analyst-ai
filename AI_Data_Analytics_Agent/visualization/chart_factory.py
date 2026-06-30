"""
Chart Factory — Interactive Plotly Chart Builder.

Unified interface for creating all chart types with consistent themes.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional

from config.constants import CHART_COLORS, PLOTLY_TEMPLATE


class ChartFactory:
    """Factory for creating themed Plotly charts."""

    @staticmethod
    def bar(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None,
            title: str = "", horizontal: bool = False) -> go.Figure:
        fn = px.bar if not horizontal else lambda **kw: px.bar(**{**kw, "orientation": "h"})
        fig = px.bar(df, x=x, y=y, color=color or x, title=title,
                     template=PLOTLY_TEMPLATE, color_discrete_sequence=CHART_COLORS)
        fig.update_layout(showlegend=bool(color), width=None)
        return fig

    @staticmethod
    def line(df: pd.DataFrame, x: str, y: str, title: str = "") -> go.Figure:
        fig = px.line(df, x=x, y=y, title=title, markers=True, template=PLOTLY_TEMPLATE)
        fig.update_traces(line=dict(width=3, color=CHART_COLORS[0]))
        return fig

    @staticmethod
    def pie(df: pd.DataFrame, names: str, values: str, title: str = "", donut: bool = False) -> go.Figure:
        fig = px.pie(df, names=names, values=values, title=title,
                     color_discrete_sequence=CHART_COLORS, hole=0.4 if donut else 0)
        return fig

    @staticmethod
    def scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None,
                title: str = "") -> go.Figure:
        fig = px.scatter(df, x=x, y=y, color=color, title=title,
                         template=PLOTLY_TEMPLATE, color_discrete_sequence=CHART_COLORS)
        return fig

    @staticmethod
    def histogram(df: pd.DataFrame, x: str, nbins: int = 30, title: str = "") -> go.Figure:
        fig = px.histogram(df, x=x, nbins=nbins, title=title,
                           template=PLOTLY_TEMPLATE, color_discrete_sequence=[CHART_COLORS[0]])
        return fig

    @staticmethod
    def box(df: pd.DataFrame, y: str, x: Optional[str] = None, title: str = "") -> go.Figure:
        fig = px.box(df, y=y, x=x, title=title, template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=CHART_COLORS)
        return fig

    @staticmethod
    def heatmap(corr_matrix: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r",
                        title=title, template=PLOTLY_TEMPLATE)
        return fig

    @staticmethod
    def treemap(df: pd.DataFrame, path: list[str], values: str, title: str = "") -> go.Figure:
        fig = px.treemap(df, path=path, values=values, title=title,
                         color_discrete_sequence=CHART_COLORS)
        return fig
