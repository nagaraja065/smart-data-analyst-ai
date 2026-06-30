"""
SQL Connector — SQLite database access via connection strings.

Provides query execution, table listing, and schema inspection for
SQLite databases using the built-in ``sqlite3`` module.

Design Pattern: Strategy — encapsulates SQL-specific loading behind
the DataConnector interface.
SOLID: Single Responsibility — only concerned with relational DB I/O.
"""

import sqlite3
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from connectors.base import DataConnector
from core.exceptions import DatabaseConnectionError, FileLoadError
from core.logger import get_logger

logger = get_logger(__name__)


class SQLConnector(DataConnector):
    """Connector for SQLite databases.

    Features
    --------
    * Execute arbitrary SELECT queries and return DataFrames.
    * List all user tables in the database.
    * Preview table schemas (column name, type, nullable, pk).
    * Connection string validation and safe resource management.
    """

    name: str = "SQL Connector"
    supported_extensions: list[str] = [".db", ".sqlite", ".sqlite3"]

    # ── Public API ───────────────────────────────────────────────────────

    def load(
        self,
        source: str,
        *,
        query: Optional[str] = None,
        table: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.

        Either *query* or *table* must be provided.  When *table* is
        given without *query*, the connector runs
        ``SELECT * FROM <table> [LIMIT <limit>]``.

        Args:
            source: Path to the SQLite database file.
            query: Raw SQL SELECT statement.
            table: Table name (used when *query* is ``None``).
            limit: Maximum rows to return.
            **kwargs: Extra arguments forwarded to ``pd.read_sql``.

        Returns:
            A pandas DataFrame with query results.

        Raises:
            DatabaseConnectionError: When the database cannot be opened
                or the query fails.
        """
        self._log_load_start(source)

        if not query and not table:
            raise DatabaseConnectionError(
                source, "Either 'query' or 'table' must be provided",
            )

        if not query:
            query = f"SELECT * FROM [{table}]"
            if limit is not None:
                query += f" LIMIT {int(limit)}"

        connection = self._get_connection(source)
        try:
            df = pd.read_sql(query, connection, **kwargs)
        except (pd.errors.DatabaseError, sqlite3.OperationalError) as exc:
            raise DatabaseConnectionError(source, f"Query failed: {exc}") from exc
        finally:
            connection.close()

        self._log_load_complete(df, source)
        return df

    def validate_source(self, source: str) -> bool:
        """Check that *source* is a valid SQLite database.

        Args:
            source: Path to the database file.

        Returns:
            ``True`` when the file exists and can be opened by SQLite.
        """
        filepath = Path(source)
        if not filepath.is_file():
            logger.warning("Database file not found: %s", source)
            return False

        # Verify SQLite can actually open the file
        try:
            conn = sqlite3.connect(f"file:{filepath}?mode=ro", uri=True)
            conn.execute("SELECT 1")
            conn.close()
            return True
        except sqlite3.Error as exc:
            logger.warning("Invalid SQLite database %s: %s", source, exc)
            return False

    def get_metadata(self, source: str, **kwargs: Any) -> dict[str, Any]:
        """Return database-level metadata.

        Args:
            source: Path to the SQLite database.

        Returns:
            Dict with ``filename``, ``size_bytes``, ``tables``, and
            per-table schemas.

        Raises:
            DatabaseConnectionError: If the database cannot be accessed.
        """
        filepath = Path(source)
        if not filepath.is_file():
            raise DatabaseConnectionError(source, "Database file not found")

        tables = self.list_tables(source)
        table_schemas: dict[str, list[dict[str, Any]]] = {}
        table_row_counts: dict[str, int] = {}

        connection = self._get_connection(source)
        try:
            for table_name in tables:
                table_schemas[table_name] = self._get_table_schema(
                    connection, table_name,
                )
                table_row_counts[table_name] = self._get_row_count(
                    connection, table_name,
                )
        finally:
            connection.close()

        return {
            "filename": filepath.name,
            "size_bytes": filepath.stat().st_size,
            "tables": tables,
            "table_count": len(tables),
            "schemas": table_schemas,
            "row_counts": table_row_counts,
        }

    def list_tables(self, source: str) -> list[str]:
        """Return all user-defined table names.

        Args:
            source: Path to the SQLite database.

        Returns:
            Sorted list of table name strings.

        Raises:
            DatabaseConnectionError: If the database cannot be read.
        """
        connection = self._get_connection(source)
        try:
            cursor = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]
        except sqlite3.OperationalError as exc:
            raise DatabaseConnectionError(
                source, f"Cannot list tables: {exc}",
            ) from exc
        finally:
            connection.close()

        logger.debug("Found %d table(s) in %s: %s", len(tables), source, tables)
        return tables

    def preview_schema(self, source: str, table: str) -> list[dict[str, Any]]:
        """Return column-level schema information for a table.

        Args:
            source: Path to the SQLite database.
            table: Table name to inspect.

        Returns:
            List of dicts, each containing ``cid``, ``name``, ``type``,
            ``notnull``, ``default_value``, and ``pk``.

        Raises:
            DatabaseConnectionError: On connection or query failure.
        """
        connection = self._get_connection(source)
        try:
            schema = self._get_table_schema(connection, table)
        finally:
            connection.close()
        return schema

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _get_connection(source: str) -> sqlite3.Connection:
        """Open a SQLite connection with safety checks.

        Args:
            source: Path to the database file.

        Returns:
            An open ``sqlite3.Connection``.

        Raises:
            DatabaseConnectionError: If the connection cannot be established.
        """
        try:
            connection = sqlite3.connect(source)
            # Enable WAL mode for better concurrent read performance
            connection.execute("PRAGMA journal_mode=WAL")
            return connection
        except sqlite3.Error as exc:
            raise DatabaseConnectionError(source, str(exc)) from exc

    @staticmethod
    def _get_table_schema(
        connection: sqlite3.Connection, table: str,
    ) -> list[dict[str, Any]]:
        """Retrieve PRAGMA table_info for *table*.

        Args:
            connection: Open SQLite connection.
            table: Table name.

        Returns:
            List of column-info dicts.
        """
        try:
            cursor = connection.execute(f"PRAGMA table_info([{table}])")
            columns = cursor.fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("Cannot read schema for table '%s': %s", table, exc)
            return []

        return [
            {
                "cid": col[0],
                "name": col[1],
                "type": col[2],
                "notnull": bool(col[3]),
                "default_value": col[4],
                "pk": bool(col[5]),
            }
            for col in columns
        ]

    @staticmethod
    def _get_row_count(connection: sqlite3.Connection, table: str) -> int:
        """Return the row count for *table*.

        Args:
            connection: Open SQLite connection.
            table: Table name.

        Returns:
            Number of rows, or -1 on error.
        """
        try:
            cursor = connection.execute(f"SELECT COUNT(*) FROM [{table}]")
            return cursor.fetchone()[0]
        except sqlite3.OperationalError as exc:
            logger.warning("Cannot count rows in '%s': %s", table, exc)
            return -1
