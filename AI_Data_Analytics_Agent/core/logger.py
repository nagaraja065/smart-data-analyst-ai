"""
Structured Logger — Enterprise Logging with File Rotation.

Provides colored console output + rotated file logs.
Every log entry includes timestamp, level, module, and message.

Design Pattern: Factory (get_logger creates configured loggers)
SOLID: Single Responsibility — only handles log setup and formatting.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config.settings import settings


# ─── Custom Formatter ────────────────────────────────────────────────────────

class ColoredFormatter(logging.Formatter):
    """Console formatter with ANSI colors for different log levels."""

    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        return super().format(record)


# ─── Logger Factory ──────────────────────────────────────────────────────────

_loggers: dict[str, logging.Logger] = {}

LOG_FORMAT = "%(asctime)s │ %(levelname)s │ %(name)-20s │ %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_DIR = settings.app.base_dir / "logs"
LOG_FILE = LOG_DIR / "app.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get or create a named logger with console + file handlers.

    Args:
        name: Logger name (usually __name__ of the calling module).
        level: Override log level. Defaults to settings.app.log_level.

    Returns:
        Configured logging.Logger instance.

    Usage:
        from core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Processing started", extra={"rows": 1000})
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    log_level = getattr(logging, (level or settings.app.log_level).upper(), logging.INFO)
    logger.setLevel(log_level)

    # Prevent duplicate handlers on reimport
    if not logger.handlers:
        # Console handler (colored)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(ColoredFormatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(console_handler)

        # File handler (rotated)
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding="utf-8"
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
            logger.addHandler(file_handler)
        except OSError:
            # If we can't write to file (permissions), continue with console only
            logger.warning("Could not create log file — using console only")

    # Don't propagate to root logger
    logger.propagate = False

    _loggers[name] = logger
    return logger
