
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
    ticker: str
    start_date: str
    end_date: str


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
            df.columns = df.columns.get_level_values(0)

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
        """
        # Step 1: check the ticker isn't blank, and make it uppercase.
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

        # Step 3a: offline mode — read CSV from the data folder.
        if self.source == "csv":
            df = self.load_from_csv(f"data/{ticker}.csv")

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
            raise ValueError(
                f"Unknown source '{self.source}'. Use 'yfinance' or 'csv'."
            )

        try:
            # yfinance does not include the end date itself.
            df = yf.download(
                ticker,
                start=query.start_date,
                end=query.end_date,
                progress=False,
                auto_adjust=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not download data for '{ticker}'. "
                f"Check your internet connection. Details: {error}"
            ) from error

        # Step 4: check if the download returned any data.
        if df is None or df.empty:
            raise ValueError(
                f"No data found for '{ticker}' between {query.start_date} and "
                f"{query.end_date}. Check the ticker symbol, the dates, "
                f"and your internet connection."
            )

        # Step 5: flatten two-level column headers.
        df = _flatten_columns(df)

        # Step 6: turn the Date index into a normal column.
        df = df.reset_index()
        df = df.rename(columns={"index": "Date", "Datetime": "Date"})

        # Step 7: keep the columns in the agreed order.
        _check_columns(df, RAW_COLUMNS)
        df = df[RAW_COLUMNS]
        df.columns.name = None

        return df

    def load_from_csv(self, path: str) -> pd.DataFrame:
        """Offline fallback so the team can work without internet."""
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find '{path}'. Put a CSV with columns "
                f"{RAW_COLUMNS} in that location."
            )
        except pd.errors.EmptyDataError:
            raise ValueError(f"'{path}' is empty.")

        # Tidy column names.
        df.columns = [str(col).strip().capitalize() for col in df.columns]

        if "Close" not in df.columns and "Adj close" in df.columns:
            df = df.rename(columns={"Adj close": "Close"})

        _check_columns(df, RAW_COLUMNS)
        df = df[RAW_COLUMNS]

        if df.empty:
            raise ValueError(f"'{path}' has column headers but no data rows.")

        return df


class DataCleaner:
    """Fixes missing values, bad rows, and date formatting."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            raise ValueError("Cannot clean an empty DataFrame. Did fetching fail?")

        _check_columns(df, RAW_COLUMNS)

        # Work on a copy.
        df = df.copy()

        # Step 1: convert the Date column to datetime.
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        if df["Date"].dt.tz is not None:
            df["Date"] = df["Date"].dt.tz_localize(None)

        df = df.dropna(subset=["Date"])

        # Step 2: convert prices and volume into numbers.
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Step 3: sort by date.
        df = df.sort_values("Date")

        # Step 4: remove duplicates.
        df = df.drop_duplicates()
        df = df.drop_duplicates(subset=["Date"], keep="last")

        # Step 5: fill missing values.
        df[["Open", "High", "Low", "Close", "Volume"]] = (
            df[["Open", "High", "Low", "Close", "Volume"]].ffill()
        )

        df = df.dropna()

        # Step 6: remove invalid closing prices.
        df = df[df["Close"] > 0]

        # Step 7: reset the index.
        df = df.reset_index(drop=True)

        if df.empty:
            raise ValueError("No usable rows left after cleaning the data.")

        return df


class FeatureBuilder:
    """Turns clean prices into model inputs."""

    def __init__(self, window: int = 5):
        self.window = window

    def add_moving_average(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        """Add a moving average column."""
        if days < 1:
            raise ValueError(
                f"Moving average days must be at least 1, got {days}."
            )

        df = df.copy()
        df[f"MA_{days}"] = df["Close"].rolling(window=days).mean()

        return df

    def add_daily_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the percentage change in closing price."""
        df = df.copy()
        df["Return"] = df["Close"].pct_change()

        return df

    def add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the next day's closing price as the target."""
        df = df.copy()
        df["Target"] = df["Close"].shift(-1)

        return df

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features and prepare data for the prediction model."""
        if df is None or df.empty:
            raise ValueError("Cannot build features from an empty DataFrame.")

        _check_columns(df, RAW_COLUMNS)

        if len(df) <= self.window:
            raise ValueError(
                f"Need more than {self.window} rows to build features, "
                f"but only got {len(df)}. Try a longer date range."
            )

        # Step 1: calculate the moving average.
        df = self.add_moving_average(df, self.window)

        # Step 2: calculate daily returns.
        df = self.add_daily_return(df)

        # Step 3: create the prediction target.
        df = self.add_target(df)

        # Step 4: FIX - preserve the latest stock data row.
        # The last row has no Target because the next closing price
        # is not known yet. Keep this row for making predictions.
        # Only remove rows without enough historical feature data.
        df = df.dropna(
            subset=[f"MA_{self.window}", "Return"]
        ).reset_index(drop=True)

        # Step 5: arrange the columns in the agreed order.
        ma_col = f"MA_{self.window}"
        df = df[RAW_COLUMNS + [ma_col, "Return", "Target"]]

        return df


if __name__ == "__main__":
    # Quick self-test using about 4 months of AAPL data.
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
        print(f"Error: {error}")
        final = None

    # Visual check of the final DataFrame.
    if final is not None:
        import matplotlib.pyplot as plt

        fig, (price_ax, return_ax) = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )

        # Chart 1: closing prices and moving average.
        price_ax.plot(
            final["Date"],
            final["Close"],
            label="Close",
            color="#1f77b4",
        )

        price_ax.plot(
            final["Date"],
            final["MA_5"],
            label="MA_5 (5-day average)",
            color="#ff7f0e",
            linestyle="--",
        )

        price_ax.set_title(
            f"{query.ticker}: Close price and 5-day moving average"
        )
        price_ax.set_ylabel("Price (USD)")
        price_ax.legend()
        price_ax.grid(alpha=0.3)

        # Chart 2: daily returns.
        colors = [
            "#2ca02c" if r >= 0 else "#d62728"
            for r in final["Return"]
        ]

        return_ax.bar(
            final["Date"],
            final["Return"] * 100,
            color=colors,
        )

        return_ax.axhline(0, color="black", linewidth=0.8)
        return_ax.set_title("Daily return")
        return_ax.set_ylabel("Return (%)")
        return_ax.grid(alpha=0.3)

        fig.autofmt_xdate()
        fig.tight_layout()

        # Save and display the chart.
        fig.savefig("data_preview.png", dpi=120)
        print("\nChart saved as data_preview.png")

        plt.show()
