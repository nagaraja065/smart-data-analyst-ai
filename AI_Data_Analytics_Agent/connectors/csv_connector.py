"""
CSV / TSV Connector — Robust flat-file ingestion with auto-detection.

Handles comma- and tab-separated files with automatic encoding detection
(utf-8 → latin-1 → cp1252 fallback), delimiter sniffing, and chunked
reading for large files.

Design Pattern: Strategy — encapsulates CSV-specific loading behind
the DataConnector interface.
SOLID: Single Responsibility — only concerned with flat-file I/O.
"""

import csv
import os
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config.constants import SUPPORTED_FILE_TYPES
from connectors.base import DataConnector
from core.exceptions import FileLoadError
from core.logger import get_logger

logger = get_logger(__name__)

# Encodings tried in order — covers ~99 % of real-world CSV files.
_ENCODINGS: list[str] = ["utf-8", "latin-1", "cp1252"]

# Files larger than this (bytes) trigger chunked reading.
_LARGE_FILE_THRESHOLD: int = 100 * 1024 * 1024  # 100 MB
_CHUNK_SIZE: int = 50_000  # rows per chunk


class CSVConnector(DataConnector):
    """Connector for CSV and TSV flat files.

    Features
    --------
    * Auto-detects encoding from a prioritized list.
    * Sniffs the delimiter from the first few lines.
    * Falls back to chunked reading for files > 100 MB.
    * Strips BOM markers and leading/trailing whitespace from headers.
    """

    name: str = "CSV Connector"
    supported_extensions: list[str] = SUPPORTED_FILE_TYPES["csv"]["extensions"]

    # ── Public API ───────────────────────────────────────────────────────

    def load(
        self,
        source: str,
        *,
        delimiter: Optional[str] = None,
        encoding: Optional[str] = None,
        chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load a CSV/TSV file into a DataFrame.

        Args:
            source: Absolute or relative path to the CSV/TSV file.
            delimiter: Explicit delimiter. ``None`` triggers auto-detection.
            encoding: Explicit encoding. ``None`` triggers auto-detection.
            chunk_size: Override the default chunk size for large files.
            **kwargs: Extra arguments forwarded to ``pd.read_csv``.

        Returns:
            A pandas DataFrame.

        Raises:
            FileLoadError: When the file is missing, unreadable, or empty.
        """
        self._log_load_start(source)
        filepath = Path(source)

        if not self.validate_source(source):
            raise FileLoadError(source, "File does not exist or is not a supported CSV/TSV file")

        resolved_encoding = encoding or self._detect_encoding(filepath)
        resolved_delimiter = delimiter or self._detect_delimiter(filepath, resolved_encoding)

        logger.debug(
            "Resolved encoding=%s, delimiter=%r for %s",
            resolved_encoding, resolved_delimiter, source,
        )

        file_size = filepath.stat().st_size
        use_chunks = file_size > _LARGE_FILE_THRESHOLD

        try:
            if use_chunks:
                df = self._read_chunked(
                    filepath,
                    resolved_encoding,
                    resolved_delimiter,
                    chunk_size or _CHUNK_SIZE,
                    **kwargs,
                )
            else:
                df = pd.read_csv(
                    filepath,
                    encoding=resolved_encoding,
                    delimiter=resolved_delimiter,
                    **kwargs,
                )
        except pd.errors.EmptyDataError as exc:
            raise FileLoadError(source, "File is empty or contains no parseable data") from exc
        except pd.errors.ParserError as exc:
            raise FileLoadError(source, f"CSV parse error: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise FileLoadError(source, f"Encoding error: {exc}") from exc
        except OSError as exc:
            raise FileLoadError(source, f"I/O error: {exc}") from exc

        # Clean up headers
        df.columns = df.columns.str.strip().str.replace(r"^\ufeff", "", regex=True)

        self._log_load_complete(df, source)
        return df

    def validate_source(self, source: str) -> bool:
        """Check that *source* exists and has a CSV/TSV extension.

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
        """Return file-level metadata without loading the full dataset.

        Args:
            source: File path.

        Returns:
            Dict with ``filename``, ``size_bytes``, ``encoding``,
            ``delimiter``, ``estimated_rows``, and ``columns``.

        Raises:
            FileLoadError: If the file cannot be accessed.
        """
        filepath = Path(source)
        if not filepath.is_file():
            raise FileLoadError(source, "File not found")

        encoding = self._detect_encoding(filepath)
        delimiter = self._detect_delimiter(filepath, encoding)

        # Read only the header + a small sample for row estimation
        try:
            sample = pd.read_csv(
                filepath, encoding=encoding, delimiter=delimiter, nrows=5,
            )
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise FileLoadError(source, f"Cannot read metadata: {exc}") from exc

        # Estimate row count from file size / average row length
        file_size = filepath.stat().st_size
        estimated_rows = self._estimate_row_count(filepath, encoding)

        return {
            "filename": filepath.name,
            "size_bytes": file_size,
            "encoding": encoding,
            "delimiter": delimiter,
            "estimated_rows": estimated_rows,
            "columns": list(sample.columns),
            "dtypes": {col: str(dtype) for col, dtype in sample.dtypes.items()},
        }

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _detect_encoding(filepath: Path) -> str:
        """Try encodings in priority order and return the first that works.

        Args:
            filepath: Path to the file to probe.

        Returns:
            A valid encoding string.

        Raises:
            FileLoadError: If no encoding can decode the file.
        """
        sample_bytes = min(8192, filepath.stat().st_size)
        raw = filepath.read_bytes()[:sample_bytes]

        for enc in _ENCODINGS:
            try:
                raw.decode(enc)
                logger.debug("Detected encoding %s for %s", enc, filepath.name)
                return enc
            except (UnicodeDecodeError, LookupError):
                continue

        raise FileLoadError(
            str(filepath),
            f"Could not detect encoding (tried {', '.join(_ENCODINGS)})",
        )

    @staticmethod
    def _detect_delimiter(filepath: Path, encoding: str) -> str:
        """Sniff the delimiter from the first few lines.

        Args:
            filepath: Path to the CSV file.
            encoding: Already-detected encoding.

        Returns:
            The detected delimiter character.
        """
        try:
            with filepath.open("r", encoding=encoding) as fh:
                sample = fh.read(8192)
        except OSError as exc:
            raise FileLoadError(str(filepath), f"Cannot read file for delimiter detection: {exc}") from exc

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            detected = dialect.delimiter
        except csv.Error:
            # Default to comma when sniffing fails
            detected = ","

        logger.debug("Detected delimiter %r for %s", detected, filepath.name)
        return detected

    @staticmethod
    def _read_chunked(
        filepath: Path,
        encoding: str,
        delimiter: str,
        chunk_size: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Read a large CSV in chunks and concatenate.

        Args:
            filepath: Path to the file.
            encoding: File encoding.
            delimiter: Column delimiter.
            chunk_size: Rows per chunk.
            **kwargs: Extra ``pd.read_csv`` arguments.

        Returns:
            The fully assembled DataFrame.
        """
        logger.info(
            "Large file detected (%s). Reading in chunks of %d rows.",
            filepath.name, chunk_size,
        )
        chunks: list[pd.DataFrame] = []
        reader = pd.read_csv(
            filepath,
            encoding=encoding,
            delimiter=delimiter,
            chunksize=chunk_size,
            **kwargs,
        )
        for i, chunk in enumerate(reader):
            chunks.append(chunk)
            if (i + 1) % 10 == 0:
                logger.debug("Read %d chunks so far", i + 1)

        return pd.concat(chunks, ignore_index=True)

    @staticmethod
    def _estimate_row_count(filepath: Path, encoding: str) -> int:
        """Estimate total rows by sampling average line length.

        Args:
            filepath: Path to the CSV file.
            encoding: File encoding.

        Returns:
            Estimated number of data rows (excluding header).
        """
        try:
            with filepath.open("r", encoding=encoding) as fh:
                sample_lines = [fh.readline() for _ in range(100)]
        except OSError:
            return -1

        sample_lines = [line for line in sample_lines if line.strip()]
        if not sample_lines:
            return 0

        avg_line_len = sum(len(line) for line in sample_lines) / len(sample_lines)
        if avg_line_len == 0:
            return 0

        file_size = filepath.stat().st_size
        # Subtract 1 for the header row
        return max(0, int(file_size / avg_line_len) - 1)
