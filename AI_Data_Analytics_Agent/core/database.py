"""
Database Manager — SQLite Session and Analysis History Storage.

Stores analysis history, user sessions, and model metadata.
Uses raw sqlite3 for zero-dependency simplicity.

Design Pattern: Repository (abstracts storage operations)
SOLID: Single Responsibility — only handles persistence.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

from core.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class DatabaseManager:
    """
    SQLite database manager for analysis history and session storage.

    Handles schema creation, CRUD operations, and connection lifecycle.
    Thread-safe for Streamlit's execution model.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file. Defaults to settings.
        """
        if db_path:
            self.db_path = Path(db_path)
        else:
            # Extract path from sqlite:///path URL
            url = settings.database.url
            self.db_path = Path(url.replace("sqlite:///", ""))

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info(f"Database initialized: {self.db_path}")

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections with auto-commit."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    parameters TEXT DEFAULT '{}',
                    results_summary TEXT,
                    rows_analyzed INTEGER DEFAULT 0,
                    columns_analyzed INTEGER DEFAULT 0,
                    duration_ms INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS model_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    dataset_name TEXT,
                    metrics TEXT DEFAULT '{}',
                    parameters TEXT DEFAULT '{}',
                    model_blob BLOB,
                    version INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS data_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    profile_data TEXT NOT NULL,
                    quality_score REAL DEFAULT 0.0,
                    row_count INTEGER DEFAULT 0,
                    column_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_analysis_session
                    ON analysis_history(session_id);
                CREATE INDEX IF NOT EXISTS idx_chat_session
                    ON chat_history(session_id);
                CREATE INDEX IF NOT EXISTS idx_model_name
                    ON model_registry(model_name);
            """)

    # ─── Analysis History ────────────────────────────────────────────────

    def save_analysis(self, session_id: str, dataset_name: str,
                      analysis_type: str, results_summary: str = "",
                      parameters: Optional[dict] = None,
                      rows: int = 0, columns: int = 0,
                      duration_ms: int = 0) -> int:
        """Save an analysis record and return its ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO analysis_history 
                   (session_id, dataset_name, analysis_type, parameters, 
                    results_summary, rows_analyzed, columns_analyzed, duration_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, dataset_name, analysis_type,
                 json.dumps(parameters or {}), results_summary,
                 rows, columns, duration_ms)
            )
            logger.info(f"Analysis saved: {analysis_type} on {dataset_name}")
            return cursor.lastrowid

    def get_analysis_history(self, session_id: Optional[str] = None,
                             limit: int = 50) -> list[dict]:
        """Retrieve analysis history, optionally filtered by session."""
        with self._get_connection() as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT * FROM analysis_history WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                    (session_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM analysis_history ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(row) for row in rows]

    # ─── Chat History ────────────────────────────────────────────────────

    def save_chat_message(self, session_id: str, role: str, content: str,
                          metadata: Optional[dict] = None) -> int:
        """Save a chat message."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO chat_history (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
                (session_id, role, content, json.dumps(metadata or {}))
            )
            return cursor.lastrowid

    def get_chat_history(self, session_id: str, limit: int = 100) -> list[dict]:
        """Get chat history for a session."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_history WHERE session_id = ? ORDER BY created_at ASC LIMIT ?",
                (session_id, limit)
            ).fetchall()
            return [dict(row) for row in rows]

    def clear_chat_history(self, session_id: str) -> None:
        """Clear chat history for a session."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))

    # ─── Model Registry ──────────────────────────────────────────────────

    def save_model(self, model_name: str, model_type: str,
                   metrics: dict, parameters: dict,
                   model_blob: Optional[bytes] = None,
                   dataset_name: str = "") -> int:
        """Save a trained model to the registry."""
        with self._get_connection() as conn:
            # Auto-increment version
            existing = conn.execute(
                "SELECT MAX(version) as max_v FROM model_registry WHERE model_name = ?",
                (model_name,)
            ).fetchone()
            version = (existing["max_v"] or 0) + 1

            cursor = conn.execute(
                """INSERT INTO model_registry 
                   (model_name, model_type, dataset_name, metrics, parameters, model_blob, version)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (model_name, model_type, dataset_name,
                 json.dumps(metrics), json.dumps(parameters),
                 model_blob, version)
            )
            logger.info(f"Model saved: {model_name} v{version}")
            return cursor.lastrowid

    def get_models(self, model_type: Optional[str] = None, limit: int = 20) -> list[dict]:
        """List models, optionally filtered by type."""
        with self._get_connection() as conn:
            if model_type:
                rows = conn.execute(
                    "SELECT id, model_name, model_type, dataset_name, metrics, version, created_at "
                    "FROM model_registry WHERE model_type = ? ORDER BY created_at DESC LIMIT ?",
                    (model_type, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, model_name, model_type, dataset_name, metrics, version, created_at "
                    "FROM model_registry ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(row) for row in rows]

    # ─── Data Profiles ───────────────────────────────────────────────────

    def save_profile(self, dataset_name: str, profile_data: dict,
                     quality_score: float, row_count: int, column_count: int) -> int:
        """Save a data profile result."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO data_profiles 
                   (dataset_name, profile_data, quality_score, row_count, column_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (dataset_name, json.dumps(profile_data), quality_score, row_count, column_count)
            )
            return cursor.lastrowid

    # ─── Utility ─────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get database usage statistics."""
        with self._get_connection() as conn:
            analysis_count = conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
            chat_count = conn.execute("SELECT COUNT(*) FROM chat_history").fetchone()[0]
            model_count = conn.execute("SELECT COUNT(*) FROM model_registry").fetchone()[0]
            profile_count = conn.execute("SELECT COUNT(*) FROM data_profiles").fetchone()[0]
            return {
                "analyses": analysis_count,
                "chat_messages": chat_count,
                "models": model_count,
                "profiles": profile_count,
            }


# ─── Module-level singleton ─────────────────────────────────────────────────

_db_instance: Optional[DatabaseManager] = None


def get_database() -> DatabaseManager:
    """Get or create the database singleton."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
