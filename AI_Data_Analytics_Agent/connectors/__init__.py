"""
Connectors Package — Unified Data Source Access Layer.

Provides a pluggable connector architecture for loading data from
CSV, Excel, SQL databases, and REST APIs into pandas DataFrames.

Design Pattern: Strategy — each connector encapsulates a specific
data-loading algorithm behind a common DataConnector interface.
SOLID: Open/Closed — add new connectors without modifying existing code.
"""

from connectors.base import DataConnector
from connectors.csv_connector import CSVConnector
from connectors.excel_connector import ExcelConnector
from connectors.sql_connector import SQLConnector
from connectors.api_connector import APIConnector

__all__ = [
    "DataConnector",
    "CSVConnector",
    "ExcelConnector",
    "SQLConnector",
    "APIConnector",
]
