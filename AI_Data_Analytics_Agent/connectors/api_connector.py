"""
REST API Connector — Fetch JSON data from HTTP endpoints.

Supports GET and POST methods with custom headers, query parameters,
authentication, and automatic JSON-to-DataFrame conversion.

Design Pattern: Strategy — encapsulates API-specific loading behind
the DataConnector interface.
SOLID: Single Responsibility — only concerned with HTTP JSON I/O.
"""

from typing import Any, Optional
from urllib.parse import urlparse

import pandas as pd
import requests

from connectors.base import DataConnector
from core.exceptions import APIConnectionError
from core.logger import get_logger

logger = get_logger(__name__)

# Default timeout in seconds for all HTTP requests.
_DEFAULT_TIMEOUT: int = 30


class APIConnector(DataConnector):
    """Connector for REST API JSON endpoints.

    Features
    --------
    * GET and POST HTTP methods.
    * Custom headers, query params, and JSON body support.
    * Configurable timeout (default 30 s).
    * Automatic JSON normalisation into flat DataFrames.
    * Supports nested JSON via a ``json_path`` key for drilling into
      the response before normalisation.
    """

    name: str = "API Connector"
    supported_extensions: list[str] = []  # URLs, not file extensions

    # ── Public API ───────────────────────────────────────────────────────

    def load(
        self,
        source: str,
        *,
        method: str = "GET",
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
        timeout: int = _DEFAULT_TIMEOUT,
        json_path: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch JSON data from a REST endpoint and return a DataFrame.

        Args:
            source: The full URL of the API endpoint.
            method: HTTP method — ``'GET'`` or ``'POST'``.
            headers: Optional HTTP headers (e.g. Authorization).
            params: Optional query-string parameters.
            json_body: Optional JSON payload for POST requests.
            timeout: Request timeout in seconds (default 30).
            json_path: Dot-separated path into the JSON response to
                extract data from (e.g. ``'data.results'``).
            **kwargs: Extra arguments forwarded to ``requests.request``.

        Returns:
            A pandas DataFrame.

        Raises:
            APIConnectionError: On network errors, timeouts, non-2xx
                status codes, or JSON parsing failures.
        """
        self._log_load_start(source)

        if not self.validate_source(source):
            raise APIConnectionError(source, reason="Invalid URL format")

        method = method.upper()
        if method not in ("GET", "POST"):
            raise APIConnectionError(
                source, reason=f"Unsupported HTTP method: {method}",
            )

        response = self._make_request(
            url=source,
            method=method,
            headers=headers,
            params=params,
            json_body=json_body,
            timeout=timeout,
            **kwargs,
        )

        data = self._parse_json(response, source, json_path)
        df = self._json_to_dataframe(data, source)

        self._log_load_complete(df, source)
        return df

    def validate_source(self, source: str) -> bool:
        """Validate that *source* is a well-formed HTTP(S) URL.

        Args:
            source: URL string.

        Returns:
            ``True`` when the URL has a valid scheme and netloc.
        """
        try:
            parsed = urlparse(source)
            is_valid = parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except ValueError:
            is_valid = False

        if not is_valid:
            logger.warning("Invalid URL: %s", source)
        return is_valid

    def get_metadata(self, source: str, **kwargs: Any) -> dict[str, Any]:
        """Return metadata about the API endpoint.

        Performs a lightweight HEAD request (falling back to a small GET)
        to collect response headers and infer content characteristics.

        Args:
            source: The API URL.

        Returns:
            Dict with ``url``, ``status_code``, ``content_type``,
            ``content_length``, and ``headers``.

        Raises:
            APIConnectionError: If the endpoint is unreachable.
        """
        if not self.validate_source(source):
            raise APIConnectionError(source, reason="Invalid URL format")

        timeout = kwargs.get("timeout", _DEFAULT_TIMEOUT)
        headers = kwargs.get("headers")

        try:
            resp = requests.head(
                source, headers=headers, timeout=timeout, allow_redirects=True,
            )
        except requests.RequestException:
            # HEAD not supported; fall back to a small GET
            try:
                resp = requests.get(
                    source, headers=headers, timeout=timeout, stream=True,
                )
            except requests.RequestException as exc:
                raise APIConnectionError(source, reason=str(exc)) from exc

        return {
            "url": source,
            "status_code": resp.status_code,
            "content_type": resp.headers.get("Content-Type", "unknown"),
            "content_length": resp.headers.get("Content-Length"),
            "headers": dict(resp.headers),
        }

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _make_request(
        url: str,
        method: str,
        headers: Optional[dict[str, str]],
        params: Optional[dict[str, Any]],
        json_body: Optional[dict[str, Any]],
        timeout: int,
        **kwargs: Any,
    ) -> requests.Response:
        """Execute the HTTP request and validate the response status.

        Args:
            url: Target URL.
            method: ``'GET'`` or ``'POST'``.
            headers: HTTP headers.
            params: Query parameters.
            json_body: JSON body payload (POST only).
            timeout: Timeout in seconds.
            **kwargs: Extra arguments for ``requests.request``.

        Returns:
            A ``requests.Response`` with a 2xx status code.

        Raises:
            APIConnectionError: On any request or response error.
        """
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=timeout,
                **kwargs,
            )
        except requests.ConnectionError as exc:
            raise APIConnectionError(url, reason=f"Connection error: {exc}") from exc
        except requests.Timeout as exc:
            raise APIConnectionError(
                url, reason=f"Request timed out after {timeout}s",
            ) from exc
        except requests.RequestException as exc:
            raise APIConnectionError(url, reason=str(exc)) from exc

        if not response.ok:
            raise APIConnectionError(
                url,
                status_code=response.status_code,
                reason=f"HTTP {response.status_code}: {response.reason}",
            )

        logger.debug(
            "API response: %d %s (%d bytes)",
            response.status_code, response.reason, len(response.content),
        )
        return response

    @staticmethod
    def _parse_json(
        response: requests.Response,
        url: str,
        json_path: Optional[str],
    ) -> Any:
        """Parse the response body as JSON and optionally drill into a path.

        Args:
            response: The HTTP response.
            url: Original URL (for error messages).
            json_path: Dot-separated key path (e.g. ``'data.results'``).

        Returns:
            The extracted JSON data (list or dict).

        Raises:
            APIConnectionError: If parsing fails or the path is invalid.
        """
        try:
            data = response.json()
        except ValueError as exc:
            raise APIConnectionError(
                url, reason=f"Response is not valid JSON: {exc}",
            ) from exc

        if json_path:
            for key in json_path.split("."):
                if isinstance(data, dict) and key in data:
                    data = data[key]
                else:
                    raise APIConnectionError(
                        url,
                        reason=(
                            f"JSON path '{json_path}' not found in response. "
                            f"Key '{key}' missing."
                        ),
                    )

        return data

    @staticmethod
    def _json_to_dataframe(data: Any, url: str) -> pd.DataFrame:
        """Convert parsed JSON data into a pandas DataFrame.

        Handles lists of records, single dicts, and nested structures
        via ``pd.json_normalize``.

        Args:
            data: Parsed JSON data.
            url: Original URL (for error messages).

        Returns:
            A pandas DataFrame.

        Raises:
            APIConnectionError: If the data cannot be converted.
        """
        try:
            if isinstance(data, list):
                if all(isinstance(item, dict) for item in data):
                    df = pd.json_normalize(data)
                else:
                    df = pd.DataFrame(data, columns=["value"])
            elif isinstance(data, dict):
                df = pd.json_normalize(data)
            else:
                raise APIConnectionError(
                    url,
                    reason=f"Unexpected JSON structure: {type(data).__name__}",
                )
        except (ValueError, TypeError) as exc:
            raise APIConnectionError(
                url, reason=f"Failed to convert JSON to DataFrame: {exc}",
            ) from exc

        if df.empty:
            logger.warning("API returned an empty dataset from %s", url)

        return df
