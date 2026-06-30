import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go


def run_forecasting(df):
    st.subheader("🔮 Sales Forecasting")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    date_col = None
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            date_col = col
            break

    if date_col and num_cols:
        _time_series_forecast(df, date_col, num_cols)
    elif num_cols:
        _regression_forecast(df, num_cols)
    else:
        st.warning("No numeric columns available for forecasting.")


def _time_series_forecast(df, date_col, num_cols):
    target = st.selectbox("Select target column to forecast", num_cols)
    model_choice = st.radio("Model", ["Linear Regression", "Random Forest"], horizontal=True)

    try:
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
        df_copy = df_copy.dropna(subset=[date_col, target])
        monthly = df_copy.groupby(df_copy[date_col].dt.to_period("M"))[target].sum().reset_index()
        monthly[date_col] = monthly[date_col].astype(str)
        monthly["month_idx"] = range(len(monthly))

        X = monthly[["month_idx"]].values
        y = monthly[target].values

        if len(X) < 4:
            st.warning("Not enough data points for forecasting (need at least 4 months).")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        if model_choice == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        col1, col2 = st.columns(2)
        col1.metric("MAE", f"₹{mae:,.0f}")
        col2.metric("R² Score", f"{r2:.3f}")

        # Forecast next 3 months
        last_idx = monthly["month_idx"].max()
        future_idx = np.array([[last_idx + i] for i in range(1, 4)])
        future_pred = model.predict(future_idx)

        st.markdown("**📅 3-Month Forecast:**")
        for i, pred in enumerate(future_pred, 1):
            st.write(f"Month +{i}: ₹{pred:,.0f}")

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly[date_col], y=monthly[target],
                                 mode="lines+markers", name="Actual", line=dict(color="#636EFA")))
        fig.add_trace(go.Scatter(x=[f"Month +{i}" for i in range(1, 4)],
                                 y=future_pred, mode="lines+markers",
                                 name="Forecast", line=dict(color="#EF553B", dash="dash")))
        fig.update_layout(title=f"{target} Forecast", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Forecasting error: {e}")


def _regression_forecast(df, num_cols):
    st.info("No date column found. Running feature-based regression.")
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
        return

    target = st.selectbox("Target column", num_cols)
    features = st.multiselect("Feature columns", [c for c in num_cols if c != target],
                              default=[c for c in num_cols if c != target][:2])

    if not features:
        return

    df_model = df[features + [target]].dropna()
    X = df_model[features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):,.2f}")
    col2.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")

    # Feature importance
    importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
    importance = importance.sort_values("Importance", ascending=False)
    st.markdown("**Feature Importance:**")
    st.dataframe(importance, use_container_width=True)
