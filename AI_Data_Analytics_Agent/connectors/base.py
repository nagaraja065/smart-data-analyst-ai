"""
Abstract Data Connector — Defines the contract all connectors must follow.

Every data source (file, database, API) is accessed through this interface,
ensuring uniform load → validate → metadata flow across the application.

Design Pattern: Template Method — subclasses implement specific loading logic
while the base class defines the workflow skeleton.
SOLID: Dependency Inversion — high-level modules depend on the DataConnector
abstraction, not on concrete CSV/Excel/SQL implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd

from core.logger import get_logger

logger = get_logger(__name__)


class DataConnector(ABC):
    """Abstract base class for all data connectors.

    Subclasses must implement ``load``, ``validate_source``, and
    ``get_metadata`` to provide a consistent interface for every
    supported data source.

    Attributes:
        name: Human-readable name of the connector (e.g. 'CSV Connector').
        supported_extensions: File extensions this connector handles.
    """

    name: str = "BaseConnector"
    supported_extensions: list[str] = []

    # ── Public contract ──────────────────────────────────────────────────

    @abstractmethod
    def load(self, source: str, **kwargs: Any) -> pd.DataFrame:
        """Load data from *source* into a DataFrame.

        Args:
            source: Path, URL, or connection string identifying the data.
            **kwargs: Connector-specific options (e.g. sheet name, query).

        Returns:
            A pandas DataFrame containing the loaded data.

        Raises:
            FileLoadError: When a file cannot be read.
            DatabaseConnectionError: When a DB connection fails.
            APIConnectionError: When an API call fails.
        """

    @abstractmethod
    def validate_source(self, source: str) -> bool:
        """Check whether *source* is reachable and well-formed.

        Args:
            source: The data-source identifier to validate.

        Returns:
            ``True`` if the source is valid and accessible.
        """

    @abstractmethod
    def get_metadata(self, source: str, **kwargs: Any) -> dict[str, Any]:
        """Return metadata about the data source without loading all rows.

        Args:
            source: The data-source identifier.
            **kwargs: Connector-specific options.

        Returns:
            A dict with keys like ``rows``, ``columns``, ``size_bytes``,
            ``schema``, etc.
        """

    # ── Shared helpers ───────────────────────────────────────────────────

    def _log_load_start(self, source: str) -> None:
        """Log the beginning of a load operation."""
        logger.info("Loading data from %s via %s", source, self.name)

    def _log_load_complete(self, df: pd.DataFrame, source: str) -> None:
        """Log successful load with row/col counts."""
        logger.info(
            "Loaded %d rows × %d columns from %s",
            len(df), len(df.columns), source,
        )

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"extensions={self.supported_extensions}>"
        )
