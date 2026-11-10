"""
PERSON 3 — APP / UI
Owns: user input, wiring everything together, charts, and output.
Uses: Person 1 (data_handler.py) and Person 2 (model.py).
"""

import sys
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from data_handler import StockQuery, DataFetcher, DataCleaner, FeatureBuilder
from model import (
    BaselinePredictor,
    LinearRegressionPredictor,
    ModelEvaluator,
    PredictionResult,
)


class InputHandler:
    """Gets and validates what the user wants."""

    def get_query(self) -> StockQuery:
        """Ask for ticker and dates using terminal input."""
        print("\n" + "=" * 60)
        print("STOCK PREDICTION APP")
        print("=" * 60)

        # Get ticker
        while True:
            ticker = input("\nEnter stock ticker (e.g., AAPL, MSFT, GOOGL): ").strip().upper()
            if ticker:
                break
            print("Ticker cannot be empty. Please try again.")

        # Get start date
        while True:
            start_date = input("Enter start date (YYYY-MM-DD): ").strip()
            try:
                pd.to_datetime(start_date)
                break
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD (e.g., 2023-01-01).")

        # Get end date
        while True:
            end_date = input("Enter end date (YYYY-MM-DD): ").strip()
            try:
                end_dt = pd.to_datetime(end_date)
                start_dt = pd.to_datetime(start_date)
                if end_dt <= start_dt:
                    print("End date must be after start date. Please try again.")
                    continue
                break
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD.")

        return StockQuery(ticker=ticker, start_date=start_date, end_date=end_date)

    def validate(self, query: StockQuery) -> bool:
        """Check ticker not empty, start < end, valid date format."""
        if not query.ticker or not query.ticker.strip():
            print("Error: Ticker cannot be empty.")
            return False

        try:
            start = pd.to_datetime(query.start_date)
            end = pd.to_datetime(query.end_date)
        except ValueError:
            print("Error: Invalid date format. Use YYYY-MM-DD.")
            return False

        if start >= end:
            print("Error: Start date must be before end date.")
            return False

        return True


class Visualizer:
    """Draws charts with matplotlib."""

    def plot_history(self, df: pd.DataFrame, ticker: str) -> None:
        """Line chart of Close price over time."""
        plt.figure(figsize=(12, 6))
        plt.plot(df["Date"], df["Close"], label="Close Price", color="#1f77b4")
        plt.title(f"{ticker} - Historical Close Prices")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_prediction(self, df: pd.DataFrame, result: PredictionResult, signal: str = None, signal_color: str = None) -> None:
        """Actual vs predicted on one chart with trading signal."""
        plt.figure(figsize=(12, 6))

        # Plot historical close prices
        plt.plot(df["Date"], df["Close"], label="Historical Close", color="#1f77b4")

        # Add the prediction point
        last_date = df["Date"].iloc[-1]
        predicted_date = pd.to_datetime(last_date) + pd.Timedelta(days=1)
        
        # Use signal color if provided, otherwise default orange
        dot_color = signal_color if signal_color else "#ff7f0e"
        plt.scatter(
            [predicted_date],
            [result.predicted_close],
            color=dot_color,
            label=f"Predicted: ${result.predicted_close:.2f}",
            s=100,
            zorder=5,
        )

        # Add signal text annotation on the chart
        if signal:
            text_color = "white" if signal in ["SELL", "BUY"] else "black"
            plt.scatter(
                [predicted_date],
                [result.predicted_close],
                marker='o',
                s=0,
                label=f"Signal: {signal}"
            )
            # Add bold signal text near the prediction point
            plt.text(
                predicted_date,
                result.predicted_close + 1,
                signal,
                ha='center',
                va='bottom',
                color=signal_color,
                fontsize=14,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
            )

        plt.title(f"{result.ticker} - Next Day Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def show_metrics(self, result: PredictionResult) -> None:
        """Print MAE and RMSE nicely."""
        print("\n" + "-" * 60)
        print("MODEL PERFORMANCE METRICS")
        print("-" * 60)
        if result.metrics:
            for metric, value in result.metrics.items():
                print(f"{metric}: {value:.4f}")
        else:
            print("No metrics available.")


class StockPredictionApp:
    """Main controller: connects data -> model -> display."""

    def __init__(self):
        self.inputs = InputHandler()
        self.fetcher = DataFetcher()
        self.cleaner = DataCleaner()
        self.features = FeatureBuilder(window=5)
        self.evaluator = ModelEvaluator()
        self.visualizer = Visualizer()

    def choose_model(self, name: str):
        """Return a predictor by name: 'baseline' or 'linear'."""
        # Model expects lowercase column names
        feature_cols = ["close", "ma_5", "return_1d"]

        if name == "baseline":
            return BaselinePredictor()
        return LinearRegressionPredictor(features=feature_cols)

    def _rename_columns_for_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename DataFrame columns to match model expectations."""
        # data_handler uses: Close, MA_5, Return, Target
        # model expects: close, ma_5, return_1d, target
        column_mapping = {
            "Close": "close",
            "MA_5": "ma_5",
            "Return": "return_1d",
            "Target": "target",
        }

        df = df.copy()
        for old, new in column_mapping.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        return df

    def _generate_signal(self, predicted_change_pct: float) -> str:
        """Generate BUY, SELL, or HOLD signal based on predicted change."""
        if predicted_change_pct > 0.5:
            return "BUY"
        elif predicted_change_pct < -0.5:
            return "SELL"
        else:
            return "HOLD"

    def _get_signal_color(self, signal: str) -> str:
        """Return color for signal visualization."""
        colors = {"BUY": "#2ca02c", "SELL": "#d62728", "HOLD": "#ff7f0e"}
        return colors.get(signal, "#808080")

    def run(self) -> None:
        """
        Main pipeline:
        1. query = self.inputs.get_query() (validate it)
        2. raw = self.fetcher.fetch(query)
        3. clean = self.cleaner.clean(raw)
        4. data = self.features.build(clean)
        5. Train model and get prediction result
        6. show charts and metrics
        """
        print("\n" + "=" * 60)
        print("STOCK PREDICTION APP")
        print("=" * 60)
        print("DISCLAIMER: This is a student project for educational purposes only.")
        print("Do NOT use for actual trading decisions.")
        print("=" * 60)

        try:
            # Step 1: Get and validate user input
            query = self.inputs.get_query()
            if not self.inputs.validate(query):
                print("\nInvalid input. Please try again.")
                return

            # Step 2: Fetch raw data
            print(f"\nFetching data for {query.ticker}...")
            try:
                raw = self.fetcher.fetch(query)
            except Exception as e:
                print(f"Failed to fetch data: {e}")
                print("\nTip: Check your internet connection or try a different ticker.")
                return

            # Step 3: Clean data
            print("Cleaning data...")
            clean = self.cleaner.clean(raw)

            # Step 4: Build features
            print("Building features...")
            data = self.features.build(clean)

            # Keep original data for plotting
            data_for_plot = data.copy()

            # Rename columns to match model expectations
            data = self._rename_columns_for_model(data)

            # Step 5: Choose model
            print("\nSelect model:")
            print("1. Baseline (tomorrow = today)")
            print("2. Linear Regression")
            model_choice = input("Enter choice (1 or 2, default 1): ").strip()

            if model_choice == "2":
                model_name = "linear"
            else:
                model_name = "baseline"

            model = self.choose_model(model_name)

            # Train and evaluate
            print(f"\nTraining {model.name} model...")
            metrics = self.evaluator.evaluate(model, data)

            # Get the prediction for the next day
            last_close = float(data["close"].iloc[-1])
            predicted_close = model.predict_next(data)

            # Create prediction result
            result = PredictionResult(
                ticker=query.ticker,
                last_close=last_close,
                predicted_close=predicted_close,
                model_name=model.name,
                metrics=metrics,
            )

            # Generate trading signal based on predicted change
            signal = self._generate_signal(result.predicted_change_pct)
            signal_color = self._get_signal_color(signal)

            # Step 6: Display results
            print("\n" + "=" * 60)
            print(f"PREDICTION RESULTS FOR {result.ticker}")
            print("=" * 60)
            print(f"Current Close Price:  ${result.last_close:.2f}")
            print(f"Predicted Next Close: ${result.predicted_close:.2f}")
            print(f"Predicted Change:     {result.predicted_change_pct:+.2f}%")
            print(f"Model Used:           {result.model_name.replace('_', ' ').title()}")
            print(f"Trading Signal:       {signal}")

            # Show metrics
            self.visualizer.show_metrics(result)

            # Show charts
            self.visualizer.plot_history(data_for_plot, query.ticker)
            self.visualizer.plot_prediction(data_for_plot, result, signal, signal_color)

            # Option to compare models
            compare = input("\nCompare all models? (y/n, default n): ").strip().lower()
            if compare in ("y", "yes"):
                print("\nComparing models...")
                try:
                    baseline = BaselinePredictor()
                    lr = LinearRegressionPredictor(features=["close", "ma_5", "return_1d"])
                    comparison = self.evaluator.compare([baseline, lr], data)
                    print("\n" + "=" * 60)
                    print("MODEL COMPARISON")
                    print("=" * 60)
                    print(comparison.to_string())
                except Exception as e:
                    print(f"Model comparison failed: {e}")

            print("\n" + "=" * 60)
            print("Done! Thanks for using Stock Prediction App.")
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please check your input and try again.")
            if "--debug" in sys.argv:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    StockPredictionApp().run()
