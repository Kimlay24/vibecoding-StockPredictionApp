"""
PERSON 3 — APP / UI
Owns: user input, wiring everything together, charts, and output.
Uses: Person 1 (data_handler.py) and Person 2 (model.py).
"""

import pandas as pd
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
        """TODO: ask for ticker and dates (input() or Streamlit widgets)."""
        raise NotImplementedError

    def validate(self, query: StockQuery) -> bool:
        """TODO: check ticker not empty, start < end, valid date format."""
        raise NotImplementedError


class Visualizer:
    """Draws charts with matplotlib (or Streamlit)."""

    def plot_history(self, df: pd.DataFrame, ticker: str) -> None:
        """TODO: line chart of Close price over time."""
        raise NotImplementedError

    def plot_prediction(self, df: pd.DataFrame, result: PredictionResult) -> None:
        """TODO: actual vs predicted on one chart."""
        raise NotImplementedError

    def show_metrics(self, result: PredictionResult) -> None:
        """TODO: print/display MAE and RMSE nicely."""
        raise NotImplementedError


class StockPredictionApp:
    """Main controller: connects data -> model -> display."""

    def __init__(self):
        self.inputs = InputHandler()
        self.fetcher = DataFetcher()
        self.cleaner = DataCleaner()
        self.features = FeatureBuilder()
        self.evaluator = ModelEvaluator()
        self.visualizer = Visualizer()

    def choose_model(self, name: str):
        """Return a predictor by name: 'baseline' or 'linear'."""
        feature_cols = ["Close", "MA_5", "Return"]  # agree on these with Person 1
        if name == "baseline":
            return BaselinePredictor(feature_cols)
        return LinearRegressionPredictor(feature_cols)

    def run(self) -> None:
        """
        TODO, in order:
        1. query = self.inputs.get_query()   (validate it)
        2. raw = self.fetcher.fetch(query)
        3. clean = self.cleaner.clean(raw)
        4. data = self.features.build(clean)
        5. result = self.evaluator.run(self.choose_model("linear"), data, query.ticker)
        6. show charts and metrics
        Add a disclaimer: this is a student project, not financial advice.
        """
        raise NotImplementedError


if __name__ == "__main__":
    StockPredictionApp().run()
