"""
Data package for Nordic Power Dashboard.

This package handles all data ingestion, storage, and orchestration:
- EntsoeAPIClient: Fetches day-ahead prices from ENTSO-E API
- PriceDatabase: Stores and queries price data in SQLite
- DataFetcher: Orchestrates the complete data pipeline
"""

from .entsoe_client import EntsoeAPIClient
from .database import PriceDatabase
from .fetcher import DataFetcher

__all__ = ['EntsoeAPIClient', 'PriceDatabase', 'DataFetcher']
