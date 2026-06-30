"""
Application Settings — Centralized Configuration Management.

Uses pydantic-style validation with python-dotenv for .env file support.
All configuration flows through this single module — no scattered os.getenv() calls.

Design Pattern: Singleton (module-level instance)
SOLID: Single Responsibility — only handles config loading and validation.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ─── Load .env file ──────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BASE_DIR / ".env")


def _get(key: str, default: str = "") -> str:
    """Get environment variable with fallback."""
    return os.getenv(key, default)


def _get_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    return _get(key, str(default)).lower() in ("true", "1", "yes")


def _get_int(key: str, default: int = 0) -> int:
    """Get integer environment variable with validation."""
    try:
        return int(_get(key, str(default)))
    except ValueError:
        return default


def _get_float(key: str, default: float = 0.0) -> float:
    """Get float environment variable with validation."""
    try:
        return float(_get(key, str(default)))
    except ValueError:
        return default


# ─── Settings Dataclasses ────────────────────────────────────────────────────

@dataclass(frozen=True)
class AppSettings:
    """Core application settings."""
    name: str = field(default_factory=lambda: _get("APP_NAME", "AI Data Analytics Agent"))
    version: str = field(default_factory=lambda: _get("APP_VERSION", "2.0.0"))
    env: str = field(default_factory=lambda: _get("APP_ENV", "development"))
    debug: bool = field(default_factory=lambda: _get_bool("DEBUG", True))
    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO"))
    base_dir: Path = field(default_factory=lambda: _BASE_DIR)

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@dataclass(frozen=True)
class LLMSettings:
    """LLM provider configuration."""
    api_key: str = field(default_factory=lambda: _get("ANTHROPIC_API_KEY", ""))
    model: str = field(default_factory=lambda: _get("LLM_MODEL", "claude-sonnet-4-20250514"))
    max_tokens: int = field(default_factory=lambda: _get_int("LLM_MAX_TOKENS", 2000))
    temperature: float = field(default_factory=lambda: _get_float("LLM_TEMPERATURE", 0.3))
    api_url: str = "https://api.anthropic.com/v1/messages"
    api_version: str = "2023-06-01"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "sk-ant-your-key-here")


@dataclass(frozen=True)
class DatabaseSettings:
    """Database configuration."""
    url: str = field(default_factory=lambda: _get("DATABASE_URL", f"sqlite:///{_BASE_DIR / 'data' / 'analytics.db'}"))

    @property
    def is_sqlite(self) -> bool:
        return self.url.startswith("sqlite")


@dataclass(frozen=True)
class PerformanceSettings:
    """Performance tuning configuration."""
    max_upload_size_mb: int = field(default_factory=lambda: _get_int("MAX_UPLOAD_SIZE_MB", 500))
    max_rows_display: int = field(default_factory=lambda: _get_int("MAX_ROWS_DISPLAY", 10000))
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 50000))
    enable_caching: bool = field(default_factory=lambda: _get_bool("ENABLE_CACHING", True))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@dataclass(frozen=True)
class ReportSettings:
    """Report generation configuration."""
    reports_dir: Path = field(default_factory=lambda: _BASE_DIR / _get("REPORTS_DIR", "reports/output"))
    charts_dir: Path = field(default_factory=lambda: _BASE_DIR / _get("CHARTS_DIR", "charts/output"))

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)


# ─── Global Settings Instance (Singleton) ────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    """Root settings container — single access point for all configuration."""
    app: AppSettings = field(default_factory=AppSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    reports: ReportSettings = field(default_factory=ReportSettings)


# Module-level singleton — import this everywhere
settings = Settings()
