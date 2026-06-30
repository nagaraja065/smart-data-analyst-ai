import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import traceback
import re
import requests
import json


class DataAgent:
    """
    AI Data Agent with multi-step reasoning, code execution, and chart generation.
    Works locally without an API key using pattern matching + pandas.
    Optionally uses Claude API for deeper analysis when an API key is provided.
    """

    def __init__(self, df, api_key=None):
        self.df = df
        self.api_key = api_key
        self.num_cols = df.select_dtypes(include="number").columns.tolist()
        self.cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.date_cols = self._detect_date_cols()
        self.max_steps = 5

    # ─── Date Detection ──────────────────────────────────────────────────────

    def _detect_date_cols(self):
        date_cols = []
        for col in self.df.columns:
            if "date" in col.lower() or "time" in col.lower():
                date_cols.append(col)
        return date_cols

    # ─── Tool Registry ───────────────────────────────────────────────────────

    def _get_tools(self):
        return {
            "run_code": {
                "description": "Execute Python code against the DataFrame (available as `df`). Use pandas, numpy, plotly.",
                "func": self._tool_run_code,
            },
            "get_column_stats": {
                "description": "Get detailed statistics for a specific column.",
                "func": self._tool_get_column_stats,
            },
            "get_sample_rows": {
                "description": "Get sample rows from the DataFrame.",
                "func": self._tool_get_sample_rows,
            },
            "get_correlation": {
                "description": "Get correlation matrix for numeric columns.",
                "func": self._tool_get_correlation,
            },
            "get_value_counts": {
                "description": "Get value counts for a categorical column.",
                "func": self._tool_get_value_counts,
            },
        }

    # ─── Tool Implementations ────────────────────────────────────────────────

    def _tool_run_code(self, code):
        """Execute pandas/plotly code in a sandboxed environment."""
        try:
            local_ns = {
                "df": self.df.copy(),
                "pd": pd,
                "np": np,
                "px": px,
                "go": go,
                "__result__": None,
                "__chart__": None,
            }
            # Wrap the last expression to capture its value
            lines = code.strip().split("\n")
            if lines:
                last_line = lines[-1].strip()
                # If last line is an expression (not assignment/import/etc), capture it
                if (not last_line.startswith(("import ", "from ", "def ", "class ", "for ", "while ", "if ", "try ", "with ", "#"))
                        and "=" not in last_line.split("(")[0]  # avoid catching function calls with =
                        or last_line.startswith("df") or last_line.startswith("result")):
                    # Try to capture result
                    exec_code = "\n".join(lines[:-1]) + f"\n__result__ = {last_line}"
                    try:
                        exec(exec_code, {"__builtins__": {}}, local_ns)
                    except:
                        exec("\n".join(lines), {"__builtins__": {}}, local_ns)
                else:
                    exec(code, {"__builtins__": {}}, local_ns)

            result = local_ns.get("__result__")
            chart = local_ns.get("__chart__")

            if result is not None:
                if isinstance(result, pd.DataFrame):
                    return {"text": result.to_string(max_rows=20), "chart": chart, "dataframe": result}
                elif isinstance(result, pd.Series):
                    return {"text": result.to_string(), "chart": chart}
                elif isinstance(result, (go.Figure,)):
                    return {"text": "Chart generated.", "chart": result}
                else:
                    return {"text": str(result), "chart": chart}
            return {"text": "Code executed successfully.", "chart": chart}
        except Exception as e:
            return {"text": f"Code Error: {e}", "chart": None}

    def _tool_get_column_stats(self, column_name):
        """Get detailed stats for a column."""
        if column_name not in self.df.columns:
            # Fuzzy match
            matches = [c for c in self.df.columns if column_name.lower() in c.lower()]
            if matches:
                column_name = matches[0]
            else:
                return {"text": f"Column '{column_name}' not found. Available: {', '.join(self.df.columns)}", "chart": None}

        col = self.df[column_name]
        stats_lines = [f"**Column: {column_name}**", f"- Type: {col.dtype}", f"- Non-null: {col.count()} / {len(col)}"]

        if pd.api.types.is_numeric_dtype(col):
            stats_lines.extend([
                f"- Mean: {col.mean():.2f}",
                f"- Median: {col.median():.2f}",
                f"- Std: {col.std():.2f}",
                f"- Min: {col.min():.2f}",
                f"- Max: {col.max():.2f}",
                f"- Sum: {col.sum():,.2f}",
            ])
        else:
            stats_lines.extend([
                f"- Unique values: {col.nunique()}",
                f"- Top value: {col.mode().iloc[0] if not col.mode().empty else 'N/A'}",
                f"- Top 5: {col.value_counts().head(5).to_dict()}",
            ])
        return {"text": "\n".join(stats_lines), "chart": None}

    def _tool_get_sample_rows(self, n=5):
        """Get sample rows."""
        n = min(int(n) if isinstance(n, (int, str)) else 5, 10)
        sample = self.df.head(n)
        return {"text": sample.to_string(), "chart": None}

    def _tool_get_correlation(self, _=None):
        """Get correlation matrix."""
        if len(self.num_cols) < 2:
            return {"text": "Need at least 2 numeric columns for correlation.", "chart": None}
        corr = self.df[self.num_cols].corr().round(3)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                        title="Correlation Heatmap", template="plotly_white")
        return {"text": corr.to_string(), "chart": fig}

    def _tool_get_value_counts(self, column_name):
        """Get value counts for a column."""
        if column_name not in self.df.columns:
            matches = [c for c in self.df.columns if column_name.lower() in c.lower()]
            if matches:
                column_name = matches[0]
            else:
                return {"text": f"Column '{column_name}' not found.", "chart": None}

        vc = self.df[column_name].value_counts().head(10)
        fig = px.bar(x=vc.index.astype(str), y=vc.values,
                     labels={"x": column_name, "y": "Count"},
                     title=f"Value Counts: {column_name}",
                     template="plotly_white",
                     color_discrete_sequence=["#636EFA"])
        return {"text": vc.to_string(), "chart": fig}

    # ─── Main Query Handler ──────────────────────────────────────────────────

    def query(self, user_question):
        """
        Main entry point. Routes to Claude (if API key) or local engine.
        Returns: {"text": str, "chart": Figure|None, "steps": list[str]}
        """
        if self.api_key:
            return self._query_with_claude(user_question)
        else:
            return self._query_local(user_question)

    # ─── Local Pattern Engine ────────────────────────────────────────────────

    def _query_local(self, question):
        """Answer questions using pattern matching + pandas. No API needed."""
        q = question.lower().strip()
        steps = ["🔍 Analyzing your question locally..."]

        # ── Dataset overview ──
        if any(kw in q for kw in ["overview", "describe", "summary", "summarize", "about", "tell me about"]):
            return self._local_overview(steps)

        # ── Anomalies / outliers ──
        if any(kw in q for kw in ["anomal", "outlier", "unusual", "abnormal"]):
            return self._local_anomalies(steps)

        # ── Top / best / highest / largest ──
        if any(kw in q for kw in ["top", "best", "highest", "largest", "most", "maximum", "max"]):
            return self._local_top(q, steps)

        # ── Bottom / worst / lowest / smallest ──
        if any(kw in q for kw in ["bottom", "worst", "lowest", "smallest", "least", "minimum", "min"]):
            return self._local_bottom(q, steps)

        # ── Total / sum ──
        if any(kw in q for kw in ["total", "sum", "overall"]):
            return self._local_total(q, steps)

        # ── Average / mean ──
        if any(kw in q for kw in ["average", "mean", "avg"]):
            return self._local_average(q, steps)

        # ── Trend / over time ──
        if any(kw in q for kw in ["trend", "over time", "monthly", "growth", "timeline"]):
            return self._local_trend(q, steps)

        # ── Distribution ──
        if any(kw in q for kw in ["distribution", "spread", "histogram"]):
            return self._local_distribution(q, steps)

        # ── Correlation ──
        if any(kw in q for kw in ["correlation", "correlate", "relationship", "related"]):
            result = self._tool_get_correlation()
            steps.append("📊 Computed correlation matrix")
            return {"text": result["text"], "chart": result["chart"], "steps": steps}

        # ── Compare ──
        if any(kw in q for kw in ["compare", "comparison", "versus", "vs", "difference"]):
            return self._local_compare(q, steps)

        # ── Count / how many ──
        if any(kw in q for kw in ["count", "how many", "number of"]):
            return self._local_count(q, steps)

        # ── Suggest charts ──
        if any(kw in q for kw in ["chart", "suggest", "visualiz", "plot", "graph"]):
            return self._local_suggest_charts(steps)

        # ── Fallback: general stats ──
        return self._local_general(q, steps)

    def _find_col_in_question(self, q, col_list=None):
        """Find which column the user is referring to."""
        cols = col_list or self.df.columns.tolist()
        # Exact match first
        for col in cols:
            if col.lower() in q:
                return col
        # Partial / word match
        for col in cols:
            words = col.lower().replace("_", " ").split()
            for word in words:
                if len(word) > 2 and word in q:
                    return col
        return None

    def _local_overview(self, steps):
        steps.append("📋 Building dataset overview")
        lines = [
            f"### 📊 Dataset Overview",
            f"- **Rows:** {len(self.df):,}",
            f"- **Columns:** {self.df.shape[1]}",
            f"- **Numeric columns:** {', '.join(self.num_cols) if self.num_cols else 'None'}",
            f"- **Categorical columns:** {', '.join(self.cat_cols) if self.cat_cols else 'None'}",
            f"- **Date columns:** {', '.join(self.date_cols) if self.date_cols else 'None'}",
            f"- **Missing values:** {int(self.df.isnull().sum().sum())}",
            "",
        ]
        if self.num_cols:
            lines.append("**Numeric Summary:**")
            for col in self.num_cols[:6]:
                lines.append(f"- **{col}**: mean={self.df[col].mean():,.2f}, "
                             f"min={self.df[col].min():,.2f}, max={self.df[col].max():,.2f}, "
                             f"total={self.df[col].sum():,.2f}")

        if self.cat_cols:
            lines.append("\n**Categorical Summary:**")
            for col in self.cat_cols[:4]:
                top = self.df[col].value_counts().head(3).to_dict()
                lines.append(f"- **{col}** ({self.df[col].nunique()} unique): Top → {top}")

        return {"text": "\n".join(lines), "chart": None, "steps": steps}

    def _local_anomalies(self, steps):
        steps.append("🔎 Detecting outliers using IQR method")
        findings = ["### 🚨 Anomaly Detection (IQR Method)\n"]
        chart = None
        for col in self.num_cols[:6]:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
            if len(outliers) > 0:
                findings.append(f"- **{col}**: {len(outliers)} outliers "
                                f"(below {lower:,.2f} or above {upper:,.2f})")

        if len(findings) == 1:
            findings.append("✅ No significant outliers found.")
        else:
            # Box plot of first few numeric columns
            cols_to_plot = self.num_cols[:4]
            fig = px.box(self.df, y=cols_to_plot, title="Outlier Distribution",
                         template="plotly_white")
            chart = fig

        return {"text": "\n".join(findings), "chart": chart, "steps": steps}

    def _local_top(self, q, steps):
        steps.append("🏆 Finding top values")
        # Extract number (default 5)
        n = 5
        nums = re.findall(r'\d+', q)
        if nums:
            n = min(int(nums[0]), 20)

        num_col = self._find_col_in_question(q, self.num_cols)
        cat_col = self._find_col_in_question(q, self.cat_cols)

        if cat_col and num_col:
            result = self.df.groupby(cat_col)[num_col].sum().nlargest(n).reset_index()
            fig = px.bar(result, x=cat_col, y=num_col,
                         title=f"Top {n} {cat_col} by {num_col}",
                         template="plotly_white", color=cat_col)
            return {"text": result.to_string(index=False), "chart": fig, "steps": steps}

        if num_col:
            result = self.df.nlargest(n, num_col)[self.df.columns[:6]]
            return {"text": f"**Top {n} by {num_col}:**\n{result.to_string(index=False)}", "chart": None, "steps": steps}

        # Default: use first numeric col, group by first cat col
        if self.cat_cols and self.num_cols:
            cat_col = self.cat_cols[0]
            num_col = self.num_cols[0]
            result = self.df.groupby(cat_col)[num_col].sum().nlargest(n).reset_index()
            fig = px.bar(result, x=cat_col, y=num_col,
                         title=f"Top {n} {cat_col} by {num_col}",
                         template="plotly_white", color=cat_col)
            return {"text": result.to_string(index=False), "chart": fig, "steps": steps}

        return {"text": "I couldn't identify which columns to use. Please specify.", "chart": None, "steps": steps}

    def _local_bottom(self, q, steps):
        steps.append("📉 Finding bottom values")
        n = 5
        nums = re.findall(r'\d+', q)
        if nums:
            n = min(int(nums[0]), 20)

        num_col = self._find_col_in_question(q, self.num_cols)
        cat_col = self._find_col_in_question(q, self.cat_cols)

        if cat_col and num_col:
            result = self.df.groupby(cat_col)[num_col].sum().nsmallest(n).reset_index()
            fig = px.bar(result, x=cat_col, y=num_col,
                         title=f"Bottom {n} {cat_col} by {num_col}",
                         template="plotly_white", color=cat_col)
            return {"text": result.to_string(index=False), "chart": fig, "steps": steps}

        if self.cat_cols and self.num_cols:
            cat_col = self.cat_cols[0]
            num_col = self.num_cols[0]
            result = self.df.groupby(cat_col)[num_col].sum().nsmallest(n).reset_index()
            fig = px.bar(result, x=cat_col, y=num_col,
                         title=f"Bottom {n} {cat_col} by {num_col}",
                         template="plotly_white", color=cat_col)
            return {"text": result.to_string(index=False), "chart": fig, "steps": steps}

        return {"text": "Please specify which columns to analyze.", "chart": None, "steps": steps}

    def _local_total(self, q, steps):
        steps.append("💰 Calculating totals")
        col = self._find_col_in_question(q, self.num_cols) or (self.num_cols[0] if self.num_cols else None)
        if col:
            total = self.df[col].sum()
            avg = self.df[col].mean()
            return {
                "text": f"### 💰 Total for **{col}**\n- **Total:** {total:,.2f}\n- **Average:** {avg:,.2f}\n- **Count:** {len(self.df):,}",
                "chart": None, "steps": steps
            }
        return {"text": "No numeric columns found to calculate totals.", "chart": None, "steps": steps}

    def _local_average(self, q, steps):
        steps.append("📊 Calculating averages")
        col = self._find_col_in_question(q, self.num_cols) or (self.num_cols[0] if self.num_cols else None)
        if col:
            cat_col = self._find_col_in_question(q, self.cat_cols)
            if cat_col:
                result = self.df.groupby(cat_col)[col].mean().round(2).reset_index()
                result.columns = [cat_col, f"Avg {col}"]
                fig = px.bar(result, x=cat_col, y=f"Avg {col}",
                             title=f"Average {col} by {cat_col}",
                             template="plotly_white", color=cat_col)
                return {"text": result.to_string(index=False), "chart": fig, "steps": steps}
            else:
                return {
                    "text": f"### 📊 Average for **{col}**\n- **Mean:** {self.df[col].mean():,.2f}\n- **Median:** {self.df[col].median():,.2f}\n- **Std Dev:** {self.df[col].std():,.2f}",
                    "chart": None, "steps": steps
                }
        return {"text": "No numeric columns found.", "chart": None, "steps": steps}

    def _local_trend(self, q, steps):
        steps.append("📈 Analyzing trends over time")
        if not self.date_cols:
            return {"text": "No date columns found for trend analysis.", "chart": None, "steps": steps}

        date_col = self.date_cols[0]
        num_col = self._find_col_in_question(q, self.num_cols) or (self.num_cols[0] if self.num_cols else None)
        if not num_col:
            return {"text": "No numeric column to plot trend.", "chart": None, "steps": steps}

        try:
            df_copy = self.df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
            monthly = df_copy.groupby(df_copy[date_col].dt.to_period("M"))[num_col].sum().reset_index()
            monthly[date_col] = monthly[date_col].astype(str)

            fig = px.line(monthly, x=date_col, y=num_col,
                          title=f"Monthly Trend: {num_col}",
                          markers=True, template="plotly_white")
            fig.update_traces(line=dict(width=3, color="#636EFA"))
            return {"text": f"Monthly trend of **{num_col}**:\n{monthly.to_string(index=False)}", "chart": fig, "steps": steps}
        except Exception as e:
            return {"text": f"Error analyzing trend: {e}", "chart": None, "steps": steps}

    def _local_distribution(self, q, steps):
        steps.append("📊 Analyzing distribution")
        col = self._find_col_in_question(q, self.num_cols) or (self.num_cols[0] if self.num_cols else None)
        if col:
            fig = px.histogram(self.df, x=col, nbins=30,
                               title=f"Distribution of {col}",
                               template="plotly_white",
                               color_discrete_sequence=["#636EFA"])
            stats = (f"**{col} Distribution:**\n"
                     f"- Mean: {self.df[col].mean():,.2f}\n"
                     f"- Median: {self.df[col].median():,.2f}\n"
                     f"- Skew: {self.df[col].skew():.2f}\n"
                     f"- Kurtosis: {self.df[col].kurtosis():.2f}")
            return {"text": stats, "chart": fig, "steps": steps}
        return {"text": "No numeric columns found.", "chart": None, "steps": steps}

    def _local_compare(self, q, steps):
        steps.append("⚖️ Comparing categories")
        cat_col = self._find_col_in_question(q, self.cat_cols) or (self.cat_cols[0] if self.cat_cols else None)
        num_col = self._find_col_in_question(q, self.num_cols) or (self.num_cols[0] if self.num_cols else None)
        if cat_col and num_col:
            result = self.df.groupby(cat_col)[num_col].agg(["sum", "mean", "count"]).round(2).reset_index()
            fig = px.bar(result, x=cat_col, y="sum",
                         title=f"Comparison: {num_col} by {cat_col}",
                         template="plotly_white", color=cat_col)
            return {"text": result.to_string(index=False), "chart": fig, "steps": steps}
        return {"text": "Please specify columns to compare.", "chart": None, "steps": steps}

    def _local_count(self, q, steps):
        steps.append("🔢 Counting values")
        cat_col = self._find_col_in_question(q, self.cat_cols) or (self.cat_cols[0] if self.cat_cols else None)
        if cat_col:
            result = self._tool_get_value_counts(cat_col)
            return {"text": f"**Counts for {cat_col}:**\n{result['text']}", "chart": result["chart"], "steps": steps}
        return {"text": f"Total rows in dataset: **{len(self.df):,}**", "chart": None, "steps": steps}

    def _local_suggest_charts(self, steps):
        steps.append("🎨 Suggesting visualizations")
        suggestions = ["### 🎨 Suggested Visualizations\n"]
        charts = []

        if self.cat_cols and self.num_cols:
            suggestions.append(f"1. **Bar Chart**: {self.cat_cols[0]} vs {self.num_cols[0]}")
            data = self.df.groupby(self.cat_cols[0])[self.num_cols[0]].sum().nlargest(8).reset_index()
            fig = px.bar(data, x=self.cat_cols[0], y=self.num_cols[0],
                         title=f"{self.num_cols[0]} by {self.cat_cols[0]}",
                         template="plotly_white", color=self.cat_cols[0])
            charts.append(fig)

        if len(self.num_cols) >= 2:
            suggestions.append(f"2. **Scatter Plot**: {self.num_cols[0]} vs {self.num_cols[1]}")
        if self.date_cols and self.num_cols:
            suggestions.append(f"3. **Line Chart**: {self.num_cols[0]} trend over {self.date_cols[0]}")
        if self.num_cols:
            suggestions.append(f"4. **Histogram**: Distribution of {self.num_cols[0]}")
        if len(self.num_cols) >= 3:
            suggestions.append(f"5. **Heatmap**: Correlation between numeric columns")

        chart = charts[0] if charts else None
        return {"text": "\n".join(suggestions), "chart": chart, "steps": steps}

    def _local_general(self, q, steps):
        """Fallback: try to find relevant columns and provide stats."""
        steps.append("🧠 Analyzing with general strategy")
        col = self._find_col_in_question(q)
        if col:
            result = self._tool_get_column_stats(col)
            return {"text": result["text"], "chart": result["chart"], "steps": steps}

        # Ultimate fallback: dataset overview
        return self._local_overview(steps)

    # ─── Claude-Powered Query ────────────────────────────────────────────────

    def _query_with_claude(self, question):
        """Use Claude API with tool-use for complex analysis."""
        steps = ["🤖 Sending to Claude AI..."]

        # Build context
        context = self._build_context_for_claude()

        prompt = f"""You are an expert data analyst AI agent. You have access to a pandas DataFrame.

Dataset Context:
{context}

User Question: {question}

Analyze the question and provide:
1. A clear, detailed answer with specific numbers
2. If applicable, key findings as bullet points
3. Strategic recommendations if relevant

Be concise but thorough. Use markdown formatting. Include specific numbers from the data."""

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1500,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )
            data = response.json()

            if "content" in data and data["content"]:
                answer = data["content"][0].get("text", "No response generated.")
                steps.append("✅ Claude response received")

                # Also try to generate a relevant chart locally
                chart = self._auto_chart_for_question(question)
                return {"text": answer, "chart": chart, "steps": steps}
            else:
                error_msg = data.get("error", {}).get("message", "Unknown API error")
                steps.append(f"❌ API Error: {error_msg}")
                # Fallback to local
                steps.append("🔄 Falling back to local analysis...")
                return self._query_local(question)

        except Exception as e:
            steps.append(f"❌ Error: {e}")
            steps.append("🔄 Falling back to local analysis...")
            return self._query_local(question)

    def _build_context_for_claude(self):
        """Build a rich context string for Claude."""
        lines = [
            f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns",
            f"Columns: {', '.join(self.df.columns)}",
            f"Numeric columns: {', '.join(self.num_cols)}",
            f"Categorical columns: {', '.join(self.cat_cols)}",
        ]

        # Numeric stats
        if self.num_cols:
            lines.append("\nNumeric Summary:")
            for col in self.num_cols[:8]:
                lines.append(f"  {col}: sum={self.df[col].sum():,.2f}, mean={self.df[col].mean():,.2f}, "
                             f"min={self.df[col].min():,.2f}, max={self.df[col].max():,.2f}")

        # Categorical top values
        for col in self.cat_cols[:4]:
            top = self.df[col].value_counts().head(5).to_dict()
            lines.append(f"\n{col} distribution: {top}")

        # Sample data
        lines.append(f"\nSample data (first 3 rows):\n{self.df.head(3).to_string()}")

        return "\n".join(lines)

    def _auto_chart_for_question(self, question):
        """Try to generate a relevant chart based on the question."""
        q = question.lower()
        try:
            if any(kw in q for kw in ["top", "best", "highest"]) and self.cat_cols and self.num_cols:
                data = self.df.groupby(self.cat_cols[0])[self.num_cols[0]].sum().nlargest(8).reset_index()
                return px.bar(data, x=self.cat_cols[0], y=self.num_cols[0],
                              template="plotly_white", color=self.cat_cols[0])
            if any(kw in q for kw in ["trend", "over time"]) and self.date_cols and self.num_cols:
                df_copy = self.df.copy()
                date_col = self.date_cols[0]
                df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
                monthly = df_copy.groupby(df_copy[date_col].dt.to_period("M"))[self.num_cols[0]].sum().reset_index()
                monthly[date_col] = monthly[date_col].astype(str)
                return px.line(monthly, x=date_col, y=self.num_cols[0],
                               markers=True, template="plotly_white")
            if any(kw in q for kw in ["distribution", "spread"]) and self.num_cols:
                return px.histogram(self.df, x=self.num_cols[0], template="plotly_white")
        except:
            pass
        return None

    # ─── Quick Actions ───────────────────────────────────────────────────────

    def quick_analyze(self):
        """Full automated analysis."""
        return self.query("Give me a complete overview and summary of this dataset")

    def quick_anomalies(self):
        """Find anomalies."""
        return self.query("Find anomalies and outliers in this dataset")

    def quick_suggest_charts(self):
        """Suggest visualizations."""
        return self.query("Suggest the best charts and visualizations for this dataset")

    def quick_insights(self):
        """Generate business insights."""
        if self.api_key:
            return self.query("Generate 6-8 actionable business insights with strategic recommendations")
        else:
            return self._generate_local_insights()

    def _generate_local_insights(self):
        """Generate insights without API."""
        steps = ["🧠 Generating insights locally..."]
        insights = ["### 📋 Auto-Generated Business Insights\n"]

        # Sales/Revenue insights
        for col in self.num_cols:
            if any(kw in col.lower() for kw in ["sales", "revenue", "amount"]):
                total = self.df[col].sum()
                avg = self.df[col].mean()
                insights.append(f"**💰 {col}:** Total = {total:,.2f}, Average = {avg:,.2f}")

                if self.cat_cols:
                    top = self.df.groupby(self.cat_cols[0])[col].sum().idxmax()
                    top_val = self.df.groupby(self.cat_cols[0])[col].sum().max()
                    insights.append(f"  - Top {self.cat_cols[0]}: **{top}** ({top_val:,.2f})")

        # Profit insights
        for col in self.num_cols:
            if any(kw in col.lower() for kw in ["profit", "margin"]):
                total = self.df[col].sum()
                neg_count = (self.df[col] < 0).sum()
                insights.append(f"\n**📈 {col}:** Total = {total:,.2f}")
                if neg_count > 0:
                    insights.append(f"  - ⚠️ {neg_count} records with negative {col}")

        # Category distribution
        if self.cat_cols:
            col = self.cat_cols[0]
            vc = self.df[col].value_counts()
            insights.append(f"\n**🗂️ {col} Distribution:**")
            for name, count in vc.head(5).items():
                pct = count / len(self.df) * 100
                insights.append(f"  - {name}: {count} ({pct:.1f}%)")

        # Recommendations
        insights.append("\n### 🎯 Recommendations")
        insights.append("- Focus on top-performing categories to maximize ROI")
        if any("profit" in c.lower() for c in self.num_cols):
            insights.append("- Investigate negative profit entries for potential loss areas")
        insights.append("- Consider time-series forecasting for future planning")
        insights.append("- Use the correlation analysis to identify key business drivers")

        return {"text": "\n".join(insights), "chart": None, "steps": steps}
