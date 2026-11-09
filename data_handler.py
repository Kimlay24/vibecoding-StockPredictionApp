"""
PERSON 1 — DATA
Owns: fetching price data, cleaning it, and building features.
Hands off: a clean pandas DataFrame to Person 2 (model.py).
"""

from dataclasses import dataclass
import pandas as pd


@dataclass
class StockQuery:
    """What the user asked for. Shared by everyone."""
    ticker: str          # e.g. "AAPL"
    start_date: str      # "YYYY-MM-DD"
    end_date: str        # "YYYY-MM-DD"


class DataFetcher:
    """Downloads raw historical prices."""

    def __init__(self, source: str = "yfinance"):
        self.source = source

    def fetch(self, query: StockQuery) -> pd.DataFrame:
        """
        Return raw OHLCV data with columns:
        Date, Open, High, Low, Close, Volume
        TODO: use yfinance (yf.download) or load a CSV from /data.
        """
        raise NotImplementedError

    def load_from_csv(self, path: str) -> pd.DataFrame:
        """Offline fallback so the team can work without internet."""
        raise NotImplementedError


class DataCleaner:
    """Fixes missing values, bad rows, and date formatting."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        TODO:
        - parse Date column, sort ascending
        - drop duplicates
        - fill or drop missing values
        """
        raise NotImplementedError


class FeatureBuilder:
    """Turns clean prices into model inputs."""

    def __init__(self, window: int = 5):
        self.window = window  # how many past days to look at

    def add_moving_average(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        """TODO: add column f'MA_{days}'."""
        raise NotImplementedError

    def add_daily_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """TODO: add column 'Return' = percent change of Close."""
        raise NotImplementedError

    def add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """TODO: add column 'Target' = next day's Close (shift -1)."""
        raise NotImplementedError

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all feature steps and drop rows with NaNs. Output goes to Person 2."""
        raise NotImplementedError
