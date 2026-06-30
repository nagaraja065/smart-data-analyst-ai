"""
Prompt Templates — Curated Prompts for Every AI Interaction.

All prompt strings live here as constants so they can be versioned, reviewed,
and A/B tested without touching agent logic.  ``format_prompt`` safely
interpolates kwargs into any template.

Design Pattern: Template Method — prompts define the skeleton; kwargs fill slots.
SOLID: Single Responsibility — prompt authoring is fully decoupled from execution.
"""

from __future__ import annotations

from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)


# ─── System-Level Prompts ────────────────────────────────────────────────────

SYSTEM_PROMPT: str = (
    "You are an expert data analyst AI agent.  You analyze datasets, generate "
    "insights, create visualizations, and answer business questions with precision.\n\n"
    "RULES:\n"
    "1. Always cite specific numbers from the data — never fabricate statistics.\n"
    "2. Use markdown formatting: headings, bullet points, bold for emphasis.\n"
    "3. When presenting numbers, use appropriate rounding and units.\n"
    "4. If data is insufficient for a conclusion, say so explicitly.\n"
    "5. Suggest follow-up analyses when relevant.\n"
    "6. For chart suggestions, specify chart type, x-axis, y-axis, and grouping."
)


# ─── Task-Specific Prompt Templates ─────────────────────────────────────────

INSIGHT_PROMPT: str = (
    "Analyze the following dataset summary and generate actionable business insights.\n\n"
    "## Dataset Overview\n"
    "{dataset_summary}\n\n"
    "## Column Statistics\n"
    "{column_stats}\n\n"
    "Provide:\n"
    "1. **Key Findings** — Top 3-5 most important observations.\n"
    "2. **Trends & Patterns** — Any notable trends, seasonality, or patterns.\n"
    "3. **Anomalies** — Unusual values or outliers worth investigating.\n"
    "4. **Recommendations** — Actionable next steps based on the data.\n"
    "5. **Suggested Visualizations** — Charts that would best illustrate findings."
)

QA_PROMPT: str = (
    "Answer the following question about the dataset using ONLY the provided data.\n\n"
    "## Question\n"
    "{question}\n\n"
    "## Dataset Info\n"
    "- Shape: {shape}\n"
    "- Columns: {columns}\n\n"
    "## Relevant Data\n"
    "{data_context}\n\n"
    "Provide a clear, concise answer with specific numbers.  If the data doesn't "
    "contain enough information to answer fully, state what is missing."
)

SUMMARY_PROMPT: str = (
    "Generate an executive summary for the following dataset.\n\n"
    "## Dataset Overview\n"
    "{dataset_summary}\n\n"
    "## Key Statistics\n"
    "{statistics}\n\n"
    "Write a professional summary covering:\n"
    "1. **Dataset Description** — What this data represents.\n"
    "2. **Size & Scope** — Scale of the dataset.\n"
    "3. **Data Quality** — Completeness, issues, concerns.\n"
    "4. **Key Metrics** — Most important numbers.\n"
    "5. **Initial Observations** — What stands out at first glance."
)

ANOMALY_PROMPT: str = (
    "Analyze the following anomaly detection results and provide interpretation.\n\n"
    "## Anomaly Report\n"
    "{anomaly_data}\n\n"
    "## Dataset Context\n"
    "{dataset_context}\n\n"
    "Provide:\n"
    "1. **Severity Assessment** — Which anomalies are most critical?\n"
    "2. **Possible Causes** — What might explain each anomaly?\n"
    "3. **Business Impact** — How could these anomalies affect operations?\n"
    "4. **Recommended Actions** — Steps to investigate or resolve."
)

FORECAST_PROMPT: str = (
    "Interpret the following time-series forecast results.\n\n"
    "## Forecast Data\n"
    "{forecast_data}\n\n"
    "## Historical Context\n"
    "{historical_context}\n\n"
    "Provide:\n"
    "1. **Trend Summary** — Overall direction and magnitude.\n"
    "2. **Confidence Assessment** — How reliable is this forecast?\n"
    "3. **Key Inflection Points** — When do significant changes occur?\n"
    "4. **Risk Factors** — What could invalidate this forecast?\n"
    "5. **Strategic Recommendations** — How to act on these predictions."
)


# ─── Template Formatter ──────────────────────────────────────────────────────

def format_prompt(template: str, **kwargs: Any) -> str:
    """Safely interpolate keyword arguments into a prompt template.

    Missing keys are left as ``{key}`` placeholders instead of raising.  Extra
    keys are silently ignored.

    Args:
        template: A string containing ``{key}`` placeholders.
        **kwargs: Values to substitute.

    Returns:
        The formatted prompt string.

    Example::

        >>> format_prompt(INSIGHT_PROMPT, dataset_summary="...", column_stats="...")
    """
    try:
        formatted = template.format(**kwargs)
        logger.debug(
            "Prompt formatted successfully — length=%d chars, keys=%s",
            len(formatted),
            list(kwargs.keys()),
        )
        return formatted
    except KeyError as exc:
        # Partial formatting: fill what we can, leave the rest as-is
        logger.warning("Prompt template missing key: %s — doing partial fill", exc)
        result = template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result
