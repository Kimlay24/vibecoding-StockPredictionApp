"""
PERSON 1 — DATA
Owns: fetching price data, cleaning it, and building features.
Hands off: a clean pandas DataFrame to Person 2 (model.py).
"""

from dataclasses import dataclass
import pandas as pd
import yfinance as yf


# The raw price columns every DataFrame from this file must have.
RAW_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


@dataclass
class StockQuery:
    """What the user asked for. Shared by everyone."""
    ticker: str          # e.g. "AAPL"
    start_date: str      # "YYYY-MM-DD"
    end_date: str        # "YYYY-MM-DD"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper: yfinance sometimes returns two rows of column headers
    (a "MultiIndex"), e.g. ("Close", "AAPL"). We keep only the level
    that holds the price names, so the columns become plain "Close", "Open", ...
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Look through each header level and keep the one that contains "Close".
        for level in range(df.columns.nlevels):
            if "Close" in df.columns.get_level_values(level):
                df.columns = df.columns.get_level_values(level)
                break
        else:
            # No level had "Close" in it, so just use the first level.
            df.columns = df.columns.get_level_values(0)
    # Remove any extra spaces around column names, e.g. " Close " -> "Close".
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _check_columns(df: pd.DataFrame, required: list) -> None:
    """Helper: stop with a clear message if any required column is missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Data is missing these columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )


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
        # Step 1: check the ticker isn't blank, and make it uppercase ("aapl" -> "AAPL").
        ticker = (query.ticker or "").strip().upper()
        if not ticker:
            raise ValueError("Ticker is empty. Please enter a symbol like 'AAPL'.")

        # Step 2: check the dates are real dates and in the right order.
        try:
            start = pd.to_datetime(query.start_date)
            end = pd.to_datetime(query.end_date)
        except (ValueError, TypeError):
            raise ValueError(
                f"Dates must look like 'YYYY-MM-DD'. Got start='{query.start_date}', "
                f"end='{query.end_date}'."
            )
        if start >= end:
            raise ValueError(
                f"Start date ({query.start_date}) must be before end date ({query.end_date})."
            )

        # Step 3a: offline mode — read the CSV from the data folder, e.g. data/AAPL.csv.
        if self.source == "csv":
            df = self.load_from_csv(f"data/{ticker}.csv")
            # Keep only the rows between the start and end dates.
            dates = pd.to_datetime(df["Date"], errors="coerce")
            df = df[(dates >= start) & (dates < end)].reset_index(drop=True)
            if df.empty:
                raise ValueError(
                    f"data/{ticker}.csv has no rows between "
                    f"{query.start_date} and {query.end_date}."
                )
            return df

        # Step 3b: online mode — download from Yahoo Finance.
        if self.source != "yfinance":
            raise ValueError(f"Unknown source '{self.source}'. Use 'yfinance' or 'csv'.")

        try:
            # Note: yfinance does NOT include the end date itself in the results.
            df = yf.download(
                ticker,
                start=query.start_date,
                end=query.end_date,
                progress=False,      # hide the download progress bar
                auto_adjust=True,    # prices adjusted for stock splits/dividends
            )
        except Exception as error:
            # Network problems, Yahoo being down, etc.
            raise RuntimeError(
                f"Could not download data for '{ticker}'. "
                f"Check your internet connection. Details: {error}"
            ) from error

        # Step 4: yfinance returns an empty table (instead of an error) for a bad
        # ticker, and also when the connection fails, so we check both cases here.
        if df is None or df.empty:
            raise ValueError(
                f"No data found for '{ticker}' between {query.start_date} and "
                f"{query.end_date}. Check the ticker symbol, the dates, "
                f"and your internet connection."
            )

        # Step 5: flatten two-level column headers into plain names.
        df = _flatten_columns(df)

        # Step 6: the dates are stored as the row labels (the "index").
        # reset_index() turns them into a normal column we can call "Date".
        df = df.reset_index()
        df = df.rename(columns={"index": "Date", "Datetime": "Date"})

        # Step 7: keep only the columns the team agreed on, in the agreed order.
        _check_columns(df, RAW_COLUMNS)
        df = df[RAW_COLUMNS]
        df.columns.name = None  # remove the leftover "Price" header label
        return df

    def load_from_csv(self, path: str) -> pd.DataFrame:
        """Offline fallback so the team can work without internet."""
        # Step 1: read the file, with a clear message if it doesn't exist.
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find '{path}'. Put a CSV with columns "
                f"{RAW_COLUMNS} in that location."
            )
        except pd.errors.EmptyDataError:
            raise ValueError(f"'{path}' is empty.")

        # Step 2: tidy the column names (strip spaces, fix lowercase like "close").
        df.columns = [str(col).strip().capitalize() for col in df.columns]

        # Step 3: if the file only has "Adj close", use it as "Close".
        if "Close" not in df.columns and "Adj close" in df.columns:
            df = df.rename(columns={"Adj close": "Close"})

        # Step 4: make sure all needed columns exist, then keep just those.
        _check_columns(df, RAW_COLUMNS)
        df = df[RAW_COLUMNS]
        if df.empty:
            raise ValueError(f"'{path}' has column headers but no data rows.")
        return df


class DataCleaner:
    """Fixes missing values, bad rows, and date formatting."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        TODO:
        - parse Date column, sort ascending
        - drop duplicates
        - fill or drop missing values
        """
        # Stop early if there's nothing to clean.
        if df is None or df.empty:
            raise ValueError("Cannot clean an empty DataFrame. Did fetching fail?")
        _check_columns(df, RAW_COLUMNS)

        # Work on a copy so we never change the original by accident.
        df = df.copy()

        # Step 1: turn the Date column into real dates.
        # errors="coerce" turns anything unreadable into NaT ("not a time").
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        # Remove timezone info if there is any, so all dates compare the same way.
        if df["Date"].dt.tz is not None:
            df["Date"] = df["Date"].dt.tz_localize(None)
        # Throw away rows where the date couldn't be read.
        df = df.dropna(subset=["Date"])

        # Step 2: make sure price and volume columns are numbers, not text.
        # Anything that isn't a number becomes NaN (missing).
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Step 3: sort oldest -> newest.
        df = df.sort_values("Date")

        # Step 4: drop duplicates — fully repeated rows, then repeated dates
        # (keep the last one, which is usually the most up to date).
        df = df.drop_duplicates()
        df = df.drop_duplicates(subset=["Date"], keep="last")

        # Step 5: fill missing values.
        # "Forward fill" copies the previous day's value into a gap,
        # which is a common way to handle a missing day in stock data.
        df[["Open", "High", "Low", "Close", "Volume"]] = (
            df[["Open", "High", "Low", "Close", "Volume"]].ffill()
        )
        # Any gaps left (e.g. at the very start, with nothing before them) get dropped.
        df = df.dropna()

        # Step 6: remove impossible rows (a price can't be zero or negative).
        df = df[df["Close"] > 0]

        # Step 7: number the rows 0, 1, 2, ... again after all the dropping.
        df = df.reset_index(drop=True)

        if df.empty:
            raise ValueError("No usable rows left after cleaning the data.")
        return df


class FeatureBuilder:
    """Turns clean prices into model inputs."""

    def __init__(self, window: int = 5):
        self.window = window  # how many past days to look at

    def add_moving_average(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        """TODO: add column f'MA_{days}'."""
        if days < 1:
            raise ValueError(f"Moving average days must be at least 1, got {days}.")
        df = df.copy()
        # Average of the last `days` closing prices (including today).
        # The first (days - 1) rows will be NaN because there isn't enough history yet.
        df[f"MA_{days}"] = df["Close"].rolling(window=days).mean()
        return df

    def add_daily_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """TODO: add column 'Return' = percent change of Close."""
        df = df.copy()
        # How much Close changed since yesterday, as a fraction:
        # 0.02 means +2%, -0.01 means -1%. The first row is NaN (no yesterday).
        df["Return"] = df["Close"].pct_change()
        return df

    def add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """TODO: add column 'Target' = next day's Close (shift -1)."""
        df = df.copy()
        # shift(-1) moves every Close up one row, so each row holds TOMORROW's price.
        # This is what the model tries to predict. The last row is NaN (no tomorrow yet).
        df["Target"] = df["Close"].shift(-1)
        return df

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all feature steps and drop rows with NaNs. Output goes to Person 2."""
        if df is None or df.empty:
            raise ValueError("Cannot build features from an empty DataFrame.")
        _check_columns(df, RAW_COLUMNS)

        # We need more rows than the window, or every row would end up NaN.
        if len(df) <= self.window:
            raise ValueError(
                f"Need more than {self.window} rows to build features, "
                f"but only got {len(df)}. Try a longer date range."
            )

        # Step 1-3: add each new column.
        df = self.add_moving_average(df, self.window)   # MA_5 when window = 5
        df = self.add_daily_return(df)
        df = self.add_target(df)

        # Step 4: drop rows with any NaN. This removes the first few rows
        # (not enough history for MA/Return) and the last row (no Target).
        df = df.dropna().reset_index(drop=True)

        # Step 5: put the columns in the order the whole team agreed on.
        ma_col = f"MA_{self.window}"
        df = df[RAW_COLUMNS + [ma_col, "Return", "Target"]]
        return df


if __name__ == "__main__":
    # Quick self-test: fetch about 4 months of AAPL and show the final table.
    today = pd.Timestamp.today().normalize()
    query = StockQuery(
        ticker="AAPL",
        start_date=(today - pd.Timedelta(days=120)).strftime("%Y-%m-%d"),
        end_date=today.strftime("%Y-%m-%d"),
    )
    print(f"Fetching {query.ticker} from {query.start_date} to {query.end_date}...")

    try:
        raw = DataFetcher().fetch(query)
        print(f"Downloaded {len(raw)} rows.")

        clean = DataCleaner().clean(raw)
        print(f"{len(clean)} rows after cleaning.")

        final = FeatureBuilder(window=5).build(clean)
        print(f"{len(final)} rows after building features.\n")

        print(final.head())
        print("\nColumns:", list(final.columns))
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        # Print a friendly message instead of a long crash report.
        print(f"Error: {error}")
        final = None

    # ---- Visual check of the final DataFrame ----
    if final is not None:
        # matplotlib is imported here (not at the top of the file) so teammates
        # who only import this file don't need it just to use the classes.
        import matplotlib.pyplot as plt

        # One figure with two charts stacked on top of each other.
        # sharex=True means both charts use the same dates along the bottom.
        fig, (price_ax, return_ax) = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},  # top chart is twice as tall
        )

        # Top chart: the daily Close price and its 5-day moving average.
        price_ax.plot(final["Date"], final["Close"], label="Close", color="#1f77b4")
        price_ax.plot(final["Date"], final["MA_5"], label="MA_5 (5-day average)",
                      color="#ff7f0e", linestyle="--")
        price_ax.set_title(f"{query.ticker}: Close price and 5-day moving average")
        price_ax.set_ylabel("Price (USD)")
        price_ax.legend()
        price_ax.grid(alpha=0.3)

        # Bottom chart: daily Return as bars. Green = price went up, red = went down.
        # We multiply by 100 so the chart shows percent (2 instead of 0.02).
        colors = ["#2ca02c" if r >= 0 else "#d62728" for r in final["Return"]]
        return_ax.bar(final["Date"], final["Return"] * 100, color=colors)
        return_ax.axhline(0, color="black", linewidth=0.8)  # the 0% line
        return_ax.set_title("Daily return")
        return_ax.set_ylabel("Return (%)")
        return_ax.grid(alpha=0.3)

        # Tilt the date labels so they don't overlap, and tidy the spacing.
        fig.autofmt_xdate()
        fig.tight_layout()

        # Save a copy as an image, then open the chart in a window.
        fig.savefig("data_preview.png", dpi=120)
        print("\nChart saved as data_preview.png")
        plt.show()