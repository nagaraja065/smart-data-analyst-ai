"""
AI Data Analytics Agent — Enterprise Dashboard.

Production-grade Streamlit application with multi-tab interface.
Integrates data profiling, EDA, KPIs, ML, AI agent, and reporting.

Version: 2.0.0
"""

import sys
import io
import uuid
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─── Project imports ─────────────────────────────────────────────────────────
from config.constants import (
    APP_TITLE, APP_ICON, COLOR_PALETTE, CHART_COLORS, PLOTLY_TEMPLATE,
    MSG_NO_DATA, ML_MODELS, COLUMN_KEYWORDS,
)
from config.settings import settings
from core.logger import get_logger
from processing.validator import DataValidator
from processing.cleaner import DataCleaner, CleaningConfig
from processing.profiler import DataProfiler
from processing.optimizer import DataOptimizer
from analytics.eda import EDAEngine
from analytics.kpi_engine import KPIEngine
from analytics.statistical import StatisticalAnalyzer
from analytics.segmentation import SegmentationEngine
from ml.predictor import MLPredictor, REGRESSION_MODELS, CLASSIFICATION_MODELS
from ml.anomaly_detector import AnomalyDetector
from ml.forecaster import TimeSeriesForecaster
from ai.agent import DataAnalyticsAgent
from visualization.chart_factory import ChartFactory

logger = get_logger(__name__)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analytics Agent",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { font-family: 'Inter', 'Segoe UI', sans-serif; }
    .main-header { 
        font-size: 2rem; font-weight: 700; 
        background: linear-gradient(90deg, #6366F1, #8B5CF6, #EC4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border-radius: 12px; padding: 1.2rem; color: white;
        border: 1px solid rgba(99,102,241,0.3);
    }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #a5b4fc; }
    .kpi-label { font-size: 0.85rem; color: #c4b5fd; }
    .quality-badge { 
        display: inline-block; padding: 4px 12px; border-radius: 20px; 
        font-weight: 600; font-size: 0.85rem;
    }
    .quality-excellent { background: #064e3b; color: #6ee7b7; }
    .quality-good { background: #1e3a5f; color: #93c5fd; }
    .quality-fair { background: #78350f; color: #fbbf24; }
    .quality-poor { background: #7f1d1d; color: #fca5a5; }
    div[data-testid="stMetric"] { 
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border: 1px solid rgba(99,102,241,0.3); border-radius: 12px;
        padding: 1rem; color: white;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State ───────────────────────────────────────────────────────────

def init_session():
    """Initialize session state."""
    defaults = {
        "df": None, "df_cleaned": None, "dataset_name": "",
        "session_id": str(uuid.uuid4())[:8],
        "chat_history": [], "api_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="main-header">📊 AI Analytics</p>', unsafe_allow_html=True)
    st.caption(f"v{settings.app.version} • Enterprise Edition")
    st.divider()

    # API Key
    api_key = st.text_input("🔑 Claude API Key", type="password", value=st.session_state.get("api_key", ""))
    if api_key:
        st.session_state["api_key"] = api_key
        st.success("✅ API Key set")

    st.divider()

    # Dataset info
    if st.session_state.df is not None:
        df = st.session_state.df
        st.success(f"📊 {st.session_state.dataset_name}")
        st.caption(f"{len(df):,} rows × {df.shape[1]} cols")
        st.caption(f"Memory: {df.memory_usage(deep=True).sum()/1024/1024:.1f} MB")
    else:
        st.info("Upload data to begin")


# ─── Helper Functions ────────────────────────────────────────────────────────

def get_df() -> pd.DataFrame:
    """Get the active DataFrame (cleaned if available)."""
    return st.session_state.get("df_cleaned") or st.session_state.get("df")

def load_sample():
    """Load the built-in sample sales dataset."""
    path = Path(__file__).parent / "data" / "sample_sales.csv"
    if path.exists():
        df = pd.read_csv(path)
        st.session_state.df = df
        st.session_state.dataset_name = "sample_sales.csv"
        return True
    return False


# ─── Tab Layout ──────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📁 Upload", "🔍 Profiling", "📊 EDA", "📈 KPIs & Viz",
    "🤖 AI Agent", "🧠 ML Studio", "🔮 Forecast", "📄 Reports"
])


# ═══════════════ TAB 1: UPLOAD ═══════════════════════════════════════════════

with tabs[0]:
    st.header("📁 Data Upload")
    st.markdown("Upload your dataset or load the built-in sample.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls", "tsv"],
                                    help="Max 500MB. Supports CSV, TSV, Excel.")
        if uploaded:
            try:
                if uploaded.name.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(uploaded)
                else:
                    df = pd.read_csv(uploaded)
                st.session_state.df = df
                st.session_state.df_cleaned = None
                st.session_state.dataset_name = uploaded.name
                st.success(f"✅ Loaded **{uploaded.name}** — {len(df):,} rows × {df.shape[1]} cols")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    with col2:
        st.markdown("### Quick Start")
        if st.button("📊 Load Sample Dataset", use_container_width=True):
            if load_sample():
                st.success("✅ Sample loaded!")
                st.rerun()

    # Preview
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("📋 Data Preview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing", f"{int(df.isnull().sum().sum()):,}")
        c4.metric("Duplicates", f"{int(df.duplicated().sum()):,}")

        st.dataframe(df.head(100), height=400)

        # Data types
        with st.expander("📊 Column Types"):
            type_df = pd.DataFrame({
                "Column": df.columns,
                "Type": [str(df[c].dtype) for c in df.columns],
                "Non-Null": [int(df[c].notna().sum()) for c in df.columns],
                "Unique": [int(df[c].nunique()) for c in df.columns],
            })
            st.dataframe(type_df, hide_index=True)


# ═══════════════ TAB 2: PROFILING ════════════════════════════════════════════

with tabs[1]:
    st.header("🔍 Data Profiling & Quality")

    if st.session_state.df is None:
        st.info(MSG_NO_DATA)
    else:
        df = st.session_state.df

        if st.button("🔍 Run Full Profile", type="primary"):
            with st.spinner("Profiling..."):
                profiler = DataProfiler()
                profile = profiler.profile(df)

                # Quality Score
                score = profile.quality_score
                badge_class = ("excellent" if score >= 90 else "good" if score >= 75
                               else "fair" if score >= 50 else "poor")
                st.markdown(f'### Data Quality: <span class="quality-badge quality-{badge_class}">'
                            f'{score}/100 — {badge_class.title()}</span>', unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows", f"{profile.row_count:,}")
                c2.metric("Columns", profile.column_count)
                c3.metric("Missing", f"{profile.total_missing:,}")
                c4.metric("Memory", f"{profile.memory_mb:.1f} MB")

                # Column profiles
                st.subheader("📋 Column Profiles")
                for cp in profile.column_profiles:
                    with st.expander(f"**{cp.name}** ({cp.dtype}) — Role: {cp.role}"):
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.write(f"Missing: **{cp.missing_count}** ({cp.missing_pct}%)")
                        mc2.write(f"Unique: **{cp.unique_count}**")
                        mc3.write(f"Role: **{cp.role}**")
                        if cp.mean is not None:
                            st.write(f"Mean: {cp.mean:,.4f} | Median: {cp.median:,.4f} | "
                                     f"Std: {cp.std:,.4f} | Skew: {cp.skew}")
                        if cp.top_values:
                            st.write("Top values:", cp.top_values)

                # Detected roles
                if profile.detected_roles:
                    st.subheader("🎯 Detected Column Roles")
                    st.json(profile.detected_roles)

        # Cleaning
        st.divider()
        st.subheader("🧹 Data Cleaning")

        with st.expander("⚙️ Cleaning Options"):
            remove_dupes = st.checkbox("Remove duplicates", True)
            fill_numeric = st.checkbox("Fill numeric nulls (median)", True)
            fill_cat = st.checkbox("Fill categorical nulls (mode)", True)
            parse_dates = st.checkbox("Parse date columns", True)
            strip_ws = st.checkbox("Strip whitespace", True)

        if st.button("🧹 Clean Data", type="primary"):
            config = CleaningConfig(
                remove_duplicates=remove_dupes, fill_numeric_nulls=fill_numeric,
                fill_categorical_nulls=fill_cat, parse_dates=parse_dates,
                strip_whitespace=strip_ws
            )
            cleaner = DataCleaner(config)
            cleaned, report = cleaner.clean(df)
            st.session_state.df_cleaned = cleaned

            st.success(f"✅ Cleaning complete!")
            for action in report.actions:
                st.write(f"  • {action}")
            st.write(f"Result: {report.rows_after:,} rows × {report.columns_after} cols")

        # Optimize
        if st.button("⚡ Optimize Memory"):
            optimizer = DataOptimizer()
            optimized, opt_report = optimizer.optimize(get_df())
            st.session_state.df_cleaned = optimized
            st.success(f"✅ {opt_report.original_memory_mb}MB → {opt_report.optimized_memory_mb}MB "
                       f"({opt_report.savings_pct}% saved)")


# ═══════════════ TAB 3: EDA ══════════════════════════════════════════════════

with tabs[2]:
    st.header("📊 Exploratory Data Analysis")

    if st.session_state.df is None:
        st.info(MSG_NO_DATA)
    else:
        df = get_df()

        if st.button("📊 Run Full EDA", type="primary"):
            with st.spinner("Analyzing..."):
                engine = EDAEngine()
                report = engine.run_full_eda(df)

                # Univariate stats
                st.subheader("📈 Univariate Analysis")
                for stat in report.univariate:
                    with st.expander(f"**{stat.column}** — μ={stat.mean:,.2f}, σ={stat.std:,.2f}"):
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Mean", f"{stat.mean:,.2f}")
                        mc2.metric("Median", f"{stat.median:,.2f}")
                        mc3.metric("Skewness", f"{stat.skewness:.2f}")
                        mc4.metric("Outliers", stat.outlier_count)

                        fig = px.histogram(df, x=stat.column, nbins=30, template=PLOTLY_TEMPLATE,
                                           color_discrete_sequence=[CHART_COLORS[0]])
                        st.plotly_chart(fig, use_container_width=True)

                # Correlation
                if report.correlation_matrix is not None:
                    st.subheader("🔗 Correlation Matrix")
                    fig = ChartFactory.heatmap(report.correlation_matrix)
                    st.plotly_chart(fig, use_container_width=True)

                    if report.top_correlations:
                        st.write("**Strongest correlations:**")
                        for pair in report.top_correlations[:5]:
                            st.write(f"  {pair.col1} ↔ {pair.col2}: **{pair.correlation:.3f}** ({pair.strength})")

                # Categorical
                if report.categorical_summaries:
                    st.subheader("🏷️ Categorical Analysis")
                    for col, info in report.categorical_summaries.items():
                        with st.expander(f"**{col}** ({info['unique']} unique)"):
                            data = df[col].value_counts().head(10).reset_index()
                            data.columns = [col, "Count"]
                            fig = px.bar(data, x=col, y="Count", template=PLOTLY_TEMPLATE,
                                         color=col, color_discrete_sequence=CHART_COLORS)
                            st.plotly_chart(fig, use_container_width=True)

                # Insights
                if report.insights:
                    st.subheader("💡 Auto-Generated Insights")
                    for insight in report.insights:
                        st.write(f"  • {insight}")


# ═══════════════ TAB 4: KPIs & VISUALIZATIONS ═══════════════════════════════

with tabs[3]:
    st.header("📈 KPIs & Visualizations")

    if st.session_state.df is None:
        st.info(MSG_NO_DATA)
    else:
        df = get_df()

        # Auto KPIs
        kpi_engine = KPIEngine()
        dashboard = kpi_engine.calculate_kpis(df)

        if dashboard.kpis:
            st.subheader("🎯 Business KPIs")
            cols = st.columns(min(4, len(dashboard.kpis)))
            for i, kpi in enumerate(dashboard.kpis[:8]):
                with cols[i % len(cols)]:
                    st.markdown(f"""<div class="kpi-card">
                        <div class="kpi-label">{kpi.icon} {kpi.name}</div>
                        <div class="kpi-value">{kpi.formatted_value}</div>
                    </div>""", unsafe_allow_html=True)

        # Interactive Chart Builder
        st.divider()
        st.subheader("🎨 Chart Builder")

        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        col1, col2, col3 = st.columns(3)
        with col1:
            chart_type = st.selectbox("Chart Type", [
                "Bar", "Line", "Pie", "Scatter", "Histogram", "Box", "Heatmap", "Treemap"
            ])
        with col2:
            x_col = st.selectbox("X Axis", df.columns)
        with col3:
            y_col = st.selectbox("Y Axis", num_cols if num_cols else df.columns)

        color_col = st.selectbox("Color (optional)", ["None"] + cat_cols)
        color_val = None if color_col == "None" else color_col

        if st.button("📊 Generate Chart", type="primary"):
            try:
                cf = ChartFactory()
                if chart_type == "Bar":
                    if x_col in cat_cols:
                        data = df.groupby(x_col)[y_col].sum().reset_index()
                        fig = cf.bar(data, x_col, y_col, title=f"{y_col} by {x_col}")
                    else:
                        fig = cf.bar(df.head(50), x_col, y_col, title=f"{y_col} by {x_col}")
                elif chart_type == "Line":
                    fig = cf.line(df, x_col, y_col, title=f"{y_col} over {x_col}")
                elif chart_type == "Pie":
                    if x_col in cat_cols:
                        data = df.groupby(x_col)[y_col].sum().reset_index()
                    else:
                        data = df.head(20)
                    fig = cf.pie(data, x_col, y_col, title=f"{y_col} distribution")
                elif chart_type == "Scatter":
                    fig = cf.scatter(df, x_col, y_col, color_val, title=f"{x_col} vs {y_col}")
                elif chart_type == "Histogram":
                    fig = cf.histogram(df, x_col, title=f"Distribution of {x_col}")
                elif chart_type == "Box":
                    fig = cf.box(df, y_col, color_val, title=f"Box Plot: {y_col}")
                elif chart_type == "Heatmap":
                    corr = df[num_cols].corr()
                    fig = cf.heatmap(corr)
                elif chart_type == "Treemap":
                    if cat_cols and num_cols:
                        data = df.groupby(cat_cols[0])[num_cols[0]].sum().reset_index()
                        fig = cf.treemap(data, [cat_cols[0]], num_cols[0], title="Treemap")
                    else:
                        st.warning("Need categorical + numeric columns")
                        fig = None

                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {e}")


# ═══════════════ TAB 5: AI AGENT ═════════════════════════════════════════════

with tabs[4]:
    st.header("🤖 AI Agent")

    if st.session_state.df is None:
        st.info(MSG_NO_DATA)
    else:
        df = get_df()
        agent = DataAnalyticsAgent(df, st.session_state.get("api_key", ""))

        # Quick Actions
        st.subheader("⚡ Quick Actions")
        qa1, qa2, qa3, qa4 = st.columns(4)
        with qa1:
            if st.button("📊 Full Analysis", use_container_width=True):
                with st.spinner("Analyzing..."):
                    resp = agent.quick_analyze()
                    st.markdown(resp.text)
                    if resp.chart:
                        st.plotly_chart(resp.chart, use_container_width=True)
        with qa2:
            if st.button("🚨 Find Anomalies", use_container_width=True):
                with st.spinner("Scanning..."):
                    resp = agent.quick_anomalies()
                    st.markdown(resp.text)
                    if resp.chart:
                        st.plotly_chart(resp.chart, use_container_width=True)
        with qa3:
            if st.button("💡 Insights", use_container_width=True):
                with st.spinner("Generating..."):
                    resp = agent.quick_insights()
                    st.markdown(resp.text)
        with qa4:
            if st.button("🎨 Suggest Charts", use_container_width=True):
                resp = agent.quick_charts()
                st.markdown(resp.text)
                if resp.chart:
                    st.plotly_chart(resp.chart, use_container_width=True)

        # Chat
        st.divider()
        st.subheader("💬 Ask Your Data")

        # Display history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input
        if prompt := st.chat_input("Ask anything about your data..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    resp = agent.query(prompt)
                    st.markdown(resp.text)
                    if resp.chart:
                        st.plotly_chart(resp.chart, use_container_width=True)
                    if resp.steps:
                        with st.expander("🔍 Reasoning Steps"):
                            for s in resp.steps:
                                st.write(s)

            st.session_state.chat_history.append({"role": "assistant", "content": resp.text})


# ═══════════════ TAB 6: ML STUDIO ════════════════════════════════════════════

with tabs[5]:
    st.header("🧠 ML Studio")

    if st.session_state.df is None:
        st.info(MSG_NO_DATA)
    else:
        df = get_df()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        all_cols = df.columns.tolist()

        ml_task = st.radio("Task Type", ["Regression", "Classification", "Anomaly Detection"], horizontal=True)

        if ml_task in ("Regression", "Classification"):
            col1, col2 = st.columns(2)
            with col1:
                target = st.selectbox("Target Variable", all_cols)
            with col2:
                available = [c for c in all_cols if c != target]
                features = st.multiselect("Feature Columns", available, default=available[:5] if len(available) >= 5 else available)

            if ml_task == "Regression":
                model_name = st.selectbox("Model", list(REGRESSION_MODELS.keys()))
            else:
                model_name = st.selectbox("Model", list(CLASSIFICATION_MODELS.keys()))

            if st.button("🚀 Train Model", type="primary"):
                if not features:
                    st.error("Select at least 1 feature")
                else:
                    with st.spinner(f"Training {model_name}..."):
                        try:
                            predictor = MLPredictor()
                            if ml_task == "Regression":
                                result = predictor.train_regression(df, target, features, model_name)
                            else:
                                result = predictor.train_classification(df, target, features, model_name)

                            st.success(f"✅ {result.model_name} trained in {result.training_time_ms}ms")

                            # Metrics
                            st.subheader("📊 Metrics")
                            metric_cols = st.columns(len(result.metrics))
                            for i, (name, val) in enumerate(result.metrics.items()):
                                metric_cols[i].metric(name, f"{val:.4f}")

                            # Feature importance
                            if result.feature_importance is not None:
                                st.subheader("🎯 Feature Importance")
                                fig = px.bar(result.feature_importance.head(10), x="Importance", y="Feature",
                                             orientation="h", template=PLOTLY_TEMPLATE,
                                             color="Importance", color_continuous_scale="Viridis")
                                st.plotly_chart(fig, use_container_width=True)

                            # Actual vs Predicted
                            if result.predictions is not None and result.actuals is not None:
                                st.subheader("📈 Actual vs Predicted")
                                comp = pd.DataFrame({"Actual": result.actuals, "Predicted": result.predictions})
                                fig = px.scatter(comp, x="Actual", y="Predicted", template=PLOTLY_TEMPLATE)
                                fig.add_trace(go.Scatter(x=[comp["Actual"].min(), comp["Actual"].max()],
                                                         y=[comp["Actual"].min(), comp["Actual"].max()],
                                                         mode="lines", name="Perfect", line=dict(dash="dash", color="red")))
                                st.plotly_chart(fig, use_container_width=True)

                            # Confusion matrix
                            if result.confusion_mat is not None:
                                st.subheader("📊 Confusion Matrix")
                                fig = px.imshow(result.confusion_mat, text_auto=True, template=PLOTLY_TEMPLATE,
                                                color_continuous_scale="Blues", title="Confusion Matrix")
                                st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"❌ Training error: {e}")

        elif ml_task == "Anomaly Detection":
            features = st.multiselect("Numeric Features", num_cols, default=num_cols[:4] if len(num_cols) >= 4 else num_cols)
            method = st.selectbox("Method", ["iqr", "zscore", "isolation_forest", "lof"])

            if st.button("🔍 Detect Anomalies", type="primary"):
                if not features:
                    st.error("Select features")
                else:
                    with st.spinner("Detecting..."):
                        detector = AnomalyDetector()
                        result = detector.detect(df, features, method)
                        st.success(f"Found **{result.anomaly_count}** anomalies ({result.anomaly_pct}%)")

                        if result.column_results:
                            for cr in result.column_results:
                                st.write(f"  • **{cr['column']}**: {cr['outliers']} outliers")

                        if result.details is not None and len(result.details) > 0:
                            st.subheader("🚨 Anomaly Details")
                            st.dataframe(result.details.head(20))

                        if features:
                            fig = px.box(df, y=features[:4], template=PLOTLY_TEMPLATE, title="Feature Distribution")
                            st.plotly_chart(fig, use_container_width=True)


# ═══════════════ TAB 7: FORECASTING ══════════════════════════════════════════

with tabs[6]:
    st.header("🔮 Time Series Forecasting")

    if st.session_state.df is None:
        st.info(MSG_NO_DATA)
    else:
        df = get_df()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        date_candidates = [c for c in df.columns if any(kw in c.lower() for kw in ["date", "time"])]

        if not date_candidates:
            st.warning("⚠️ No date column detected. Forecasting requires a date column.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                date_col = st.selectbox("Date Column", date_candidates)
            with col2:
                value_col = st.selectbox("Value to Forecast", num_cols)
            with col3:
                periods = st.slider("Forecast Periods", 1, 12, 3)

            method = st.radio("Method", ["linear", "moving_average", "random_forest"], horizontal=True)

            if st.button("🔮 Generate Forecast", type="primary"):
                with st.spinner("Forecasting..."):
                    try:
                        forecaster = TimeSeriesForecaster()
                        result = forecaster.forecast(df, date_col, value_col, periods, method)

                        # Metrics
                        st.subheader("📊 Model Metrics")
                        mc = st.columns(3)
                        mc[0].metric("MAE", f"{result.metrics['MAE']:,.2f}")
                        mc[1].metric("RMSE", f"{result.metrics['RMSE']:,.2f}")
                        mc[2].metric("R²", f"{result.metrics['R²']:.4f}")

                        # Historical + Forecast chart
                        st.subheader("📈 Forecast")
                        hist = result.historical[["Period_str", "Value"]].copy()
                        hist.columns = ["Period", "Value"]
                        hist["Type"] = "Historical"

                        fc = result.forecast[["Period", "Forecast"]].copy()
                        fc.columns = ["Period", "Value"]
                        fc["Type"] = "Forecast"

                        combined = pd.concat([hist, fc], ignore_index=True)
                        fig = px.line(combined, x="Period", y="Value", color="Type", markers=True,
                                      template=PLOTLY_TEMPLATE, color_discrete_map={"Historical": CHART_COLORS[0], "Forecast": CHART_COLORS[3]})
                        st.plotly_chart(fig, use_container_width=True)

                        # Forecast table
                        st.subheader("📋 Forecast Details")
                        st.dataframe(result.forecast, hide_index=True)

                    except Exception as e:
                        st.error(f"❌ Forecast error: {e}")


# ═══════════════ TAB 8: REPORTS ══════════════════════════════════════════════

with tabs[7]:
    st.header("📄 Reports")

    if st.session_state.df is None:
        st.info(MSG_NO_DATA)
    else:
        df = get_df()

        st.subheader("📥 Download Reports")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Excel Report")
            if st.button("📥 Generate Excel", use_container_width=True):
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Raw Data", index=False)
                    df.describe().to_excel(writer, sheet_name="Statistics")
                    num = df.select_dtypes(include="number")
                    if len(num.columns) >= 2:
                        num.corr().to_excel(writer, sheet_name="Correlations")
                    cat = df.select_dtypes(include=["object", "category"]).columns
                    if len(cat) > 0 and len(num.columns) > 0:
                        summary = df.groupby(cat[0])[num.columns[0]].agg(["sum", "mean", "count"])
                        summary.to_excel(writer, sheet_name="Category Summary")
                buffer.seek(0)
                st.download_button("📥 Download Excel", buffer, f"{st.session_state.dataset_name}_report.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        with col2:
            st.markdown("### 📋 CSV Export")
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, f"{st.session_state.dataset_name}_export.csv",
                               "text/csv", use_container_width=True)

        # Summary stats
        st.divider()
        st.subheader("📊 Executive Summary")
        kpi_engine = KPIEngine()
        dashboard = kpi_engine.calculate_kpis(df)
        if dashboard.kpis:
            for kpi in dashboard.kpis[:10]:
                st.write(f"{kpi.icon} **{kpi.name}**: {kpi.formatted_value}")

        # Statistical summary
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe().round(2), use_container_width=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(f"📊 AI Data Analytics Agent v{settings.app.version} • Enterprise Edition • Built with Streamlit + Plotly + scikit-learn")
