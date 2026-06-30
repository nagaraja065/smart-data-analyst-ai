"""
Custom Exception Hierarchy — Structured Error Handling.

Every exception carries an error code, context, and user-friendly message.
No bare 'except Exception' anywhere — catch specific types.

Design Pattern: Inheritance hierarchy
SOLID: Liskov Substitution — all exceptions are interchangeable where base is expected.
"""

from typing import Any, Optional


class AnalyticsBaseError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, code: str = "ERR_UNKNOWN",
                 details: Optional[dict[str, Any]] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


# ─── Data Layer Exceptions ───────────────────────────────────────────────────

class DataError(AnalyticsBaseError):
    """Base for all data-related errors."""

    def __init__(self, message: str, code: str = "ERR_DATA", **kwargs):
        super().__init__(message, code, **kwargs)


class FileLoadError(DataError):
    """Failed to load a file (corrupt, wrong format, permissions)."""

    def __init__(self, filepath: str, reason: str = ""):
        details = {"filepath": filepath}
        msg = f"Failed to load file: {filepath}"
        if reason:
            msg += f" — {reason}"
        super().__init__(msg, code="ERR_FILE_LOAD", details=details)


class UnsupportedFormatError(DataError):
    """File format not supported."""

    def __init__(self, extension: str, supported: list[str]):
        super().__init__(
            f"Unsupported file format: '{extension}'. Supported: {', '.join(supported)}",
            code="ERR_UNSUPPORTED_FORMAT",
            details={"extension": extension, "supported": supported}
        )


class DataValidationError(DataError):
    """Data failed validation checks."""

    def __init__(self, issues: list[str]):
        super().__init__(
            f"Data validation failed with {len(issues)} issue(s)",
            code="ERR_VALIDATION",
            details={"issues": issues}
        )


class EmptyDatasetError(DataError):
    """Dataset has no rows after filtering/cleaning."""

    def __init__(self, context: str = ""):
        msg = "Dataset is empty"
        if context:
            msg += f" after {context}"
        super().__init__(msg, code="ERR_EMPTY_DATASET")


class ColumnNotFoundError(DataError):
    """Referenced column doesn't exist in DataFrame."""

    def __init__(self, column: str, available: list[str]):
        super().__init__(
            f"Column '{column}' not found. Available: {', '.join(available[:10])}",
            code="ERR_COLUMN_NOT_FOUND",
            details={"column": column, "available": available}
        )


# ─── Connection Exceptions ──────────────────────────────────────────────────

class ConnectionError(AnalyticsBaseError):
    """Base for connection-related errors."""

    def __init__(self, message: str, code: str = "ERR_CONNECTION", **kwargs):
        super().__init__(message, code, **kwargs)


class DatabaseConnectionError(ConnectionError):
    """Failed to connect to database."""

    def __init__(self, db_url: str, reason: str = ""):
        msg = f"Database connection failed: {db_url}"
        if reason:
            msg += f" — {reason}"
        super().__init__(msg, code="ERR_DB_CONNECTION", details={"db_url": db_url})


class APIConnectionError(ConnectionError):
    """Failed to connect to external API."""

    def __init__(self, url: str, status_code: Optional[int] = None, reason: str = ""):
        details = {"url": url}
        if status_code:
            details["status_code"] = status_code
        msg = f"API connection failed: {url}"
        if reason:
            msg += f" — {reason}"
        super().__init__(msg, code="ERR_API_CONNECTION", details=details)


# ─── ML Exceptions ──────────────────────────────────────────────────────────

class MLError(AnalyticsBaseError):
    """Base for ML-related errors."""

    def __init__(self, message: str, code: str = "ERR_ML", **kwargs):
        super().__init__(message, code, **kwargs)


class InsufficientDataError(MLError):
    """Not enough data for ML operation."""

    def __init__(self, required: int, actual: int, operation: str = "training"):
        super().__init__(
            f"Insufficient data for {operation}: need {required}, got {actual}",
            code="ERR_INSUFFICIENT_DATA",
            details={"required": required, "actual": actual, "operation": operation}
        )


class ModelTrainingError(MLError):
    """Model training failed."""

    def __init__(self, model_name: str, reason: str = ""):
        msg = f"Training failed for model: {model_name}"
        if reason:
            msg += f" — {reason}"
        super().__init__(msg, code="ERR_TRAINING", details={"model": model_name})


# ─── AI Agent Exceptions ────────────────────────────────────────────────────

class AgentError(AnalyticsBaseError):
    """Base for AI agent errors."""

    def __init__(self, message: str, code: str = "ERR_AGENT", **kwargs):
        super().__init__(message, code, **kwargs)


class LLMProviderError(AgentError):
    """LLM provider returned an error."""

    def __init__(self, provider: str, reason: str = ""):
        msg = f"LLM provider error ({provider})"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, code="ERR_LLM_PROVIDER", details={"provider": provider})


class ToolExecutionError(AgentError):
    """Agent tool execution failed."""

    def __init__(self, tool_name: str, reason: str = ""):
        msg = f"Tool execution failed: {tool_name}"
        if reason:
            msg += f" — {reason}"
        super().__init__(msg, code="ERR_TOOL_EXEC", details={"tool": tool_name})


# ─── Report Exceptions ──────────────────────────────────────────────────────

class ReportError(AnalyticsBaseError):
    """Report generation failed."""

    def __init__(self, report_type: str, reason: str = ""):
        msg = f"Report generation failed ({report_type})"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, code="ERR_REPORT", details={"report_type": report_type})


# ─── Configuration Exceptions ───────────────────────────────────────────────

class ConfigurationError(AnalyticsBaseError):
    """Application misconfigured."""

    def __init__(self, setting: str, reason: str = ""):
        msg = f"Configuration error: {setting}"
        if reason:
            msg += f" — {reason}"
        super().__init__(msg, code="ERR_CONFIG", details={"setting": setting})
