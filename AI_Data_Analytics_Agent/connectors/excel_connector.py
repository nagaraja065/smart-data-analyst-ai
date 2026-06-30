"""
Excel Connector — .xlsx / .xls workbook ingestion.

Supports sheet selection, sheet listing, and metadata extraction using
the openpyxl engine for modern Excel files and xlrd for legacy .xls.

Design Pattern: Strategy — encapsulates Excel-specific loading behind
the DataConnector interface.
SOLID: Single Responsibility — only concerned with Excel workbook I/O.
"""

from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from config.constants import SUPPORTED_FILE_TYPES
from connectors.base import DataConnector
from core.exceptions import FileLoadError
from core.logger import get_logger

logger = get_logger(__name__)


class ExcelConnector(DataConnector):
    """Connector for Microsoft Excel workbooks (.xlsx, .xls).

    Features
    --------
    * Automatic engine selection (openpyxl for .xlsx, xlrd for .xls).
    * Load a specific sheet by name or index.
    * List all available sheet names.
    * Extract metadata (sheets, dimensions, column types) without
      reading the full dataset.
    """

    name: str = "Excel Connector"
    supported_extensions: list[str] = SUPPORTED_FILE_TYPES["excel"]["extensions"]

    # ── Public API ───────────────────────────────────────────────────────

    def load(
        self,
        source: str,
        *,
        sheet_name: Optional[Union[str, int]] = 0,
        header: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load an Excel worksheet into a DataFrame.

        Args:
            source: Path to the .xlsx / .xls file.
            sheet_name: Sheet name (str) or zero-based index (int).
                Defaults to the first sheet.
            header: Row number to use as column headers.
            **kwargs: Extra arguments forwarded to ``pd.read_excel``.

        Returns:
            A pandas DataFrame.

        Raises:
            FileLoadError: When the file is missing, corrupt, or the
                requested sheet does not exist.
        """
        self._log_load_start(source)
        filepath = Path(source)

        if not self.validate_source(source):
            raise FileLoadError(source, "File does not exist or is not a supported Excel file")

        engine = self._resolve_engine(filepath)

        try:
            df = pd.read_excel(
                filepath,
                sheet_name=sheet_name,
                header=header,
                engine=engine,
                **kwargs,
            )
        except ValueError as exc:
            raise FileLoadError(
                source, f"Sheet not found or invalid parameter: {exc}",
            ) from exc
        except ImportError as exc:
            raise FileLoadError(
                source,
                f"Missing engine dependency: {exc}. "
                "Install openpyxl (pip install openpyxl) or xlrd.",
            ) from exc
        except OSError as exc:
            raise FileLoadError(source, f"I/O error: {exc}") from exc

        # Clean up headers
        df.columns = df.columns.astype(str).str.strip()

        self._log_load_complete(df, source)
        return df

    def validate_source(self, source: str) -> bool:
        """Check that *source* exists and has an Excel extension.

        Args:
            source: File path to validate.

        Returns:
            ``True`` when the path exists and the extension is supported.
        """
        filepath = Path(source)
        if not filepath.is_file():
            logger.warning("Source file not found: %s", source)
            return False
        if filepath.suffix.lower() not in self.supported_extensions:
            logger.warning(
                "Unsupported extension %s for %s", filepath.suffix, self.name,
            )
            return False
        return True

    def get_metadata(self, source: str, **kwargs: Any) -> dict[str, Any]:
        """Return workbook-level metadata without loading all data.

        Args:
            source: Path to the Excel file.

        Returns:
            Dict with ``filename``, ``size_bytes``, ``engine``,
            ``sheets``, and per-sheet column previews.

        Raises:
            FileLoadError: If the file cannot be accessed.
        """
        filepath = Path(source)
        if not filepath.is_file():
            raise FileLoadError(source, "File not found")

        engine = self._resolve_engine(filepath)
        sheets = self.list_sheets(source)

        # Preview first sheet's columns and row count
        sheet_previews: dict[str, Any] = {}
        for sheet in sheets:
            try:
                sample = pd.read_excel(
                    filepath, sheet_name=sheet, engine=engine, nrows=5,
                )
                sheet_previews[sheet] = {
                    "columns": list(sample.columns),
                    "dtypes": {col: str(dtype) for col, dtype in sample.dtypes.items()},
                    "preview_rows": len(sample),
                }
            except (ValueError, OSError) as exc:
                logger.warning("Could not preview sheet '%s': %s", sheet, exc)
                sheet_previews[sheet] = {"error": str(exc)}

        return {
            "filename": filepath.name,
            "size_bytes": filepath.stat().st_size,
            "engine": engine,
            "sheets": sheets,
            "sheet_count": len(sheets),
            "sheet_previews": sheet_previews,
        }

    def list_sheets(self, source: str) -> list[str]:
        """Return a list of sheet names in the workbook.

        Args:
            source: Path to the Excel file.

        Returns:
            List of sheet name strings.

        Raises:
            FileLoadError: If the file cannot be read.
        """
        filepath = Path(source)
        engine = self._resolve_engine(filepath)

        try:
            xlsx = pd.ExcelFile(filepath, engine=engine)
            sheet_names: list[str] = xlsx.sheet_names
            xlsx.close()
        except (ValueError, ImportError, OSError) as exc:
            raise FileLoadError(
                source, f"Cannot read sheet list: {exc}",
            ) from exc

        logger.debug("Found %d sheet(s) in %s: %s", len(sheet_names), filepath.name, sheet_names)
        return sheet_names

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_engine(filepath: Path) -> str:
        """Select the appropriate pandas Excel engine.

        Args:
            filepath: Path to the Excel file.

        Returns:
            ``'openpyxl'`` for .xlsx, ``'xlrd'`` for .xls.
        """
        suffix = filepath.suffix.lower()
        if suffix == ".xls":
            return "xlrd"
        return "openpyxl"
