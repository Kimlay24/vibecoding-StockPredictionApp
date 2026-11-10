"""
model.py  --  Person 2 (Model)

Contains:
  - PredictionResult : what the model hands to the app/UI
  - BasePredictor    : shared interface every model must follow
  - BaselinePredictor: "tomorrow = today" benchmark
  - LinearRegressionPredictor: simple scikit-learn model
  - ModelEvaluator   : time-based split + MAE / RMSE

Expected input (from data_handler.py): a pandas DataFrame sorted by date with
  - feature columns (default: ma_5, ma_20, return_1d, close)
  - a "close" column (today's closing price)
  - a "target" column (next day's closing price)
Agree on these names with your teammates on day one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DEFAULT_FEATURES: List[str] = ["close", "ma_5", "ma_20", "return_1d"]
TARGET_COL = "target"


# --------------------------------------------------------------------------- #
# Result object shared with the app/UI
# --------------------------------------------------------------------------- #
@dataclass
class PredictionResult:
    ticker: str
    last_close: float
    predicted_close: float
    model_name: str
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def predicted_change_pct(self) -> float:
        return (self.predicted_close - self.last_close) / self.last_close * 100


# --------------------------------------------------------------------------- #
# Predictor interface
# --------------------------------------------------------------------------- #
class BasePredictor(ABC):
    """All models implement fit() and predict()."""

    name: str = "base"

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BasePredictor":
        """Train on a DataFrame that contains features and the target column."""

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return one predicted next-day close per row of df."""

    def predict_next(self, df: pd.DataFrame) -> float:
        """Predict tomorrow's close from the most recent row."""
        return float(self.predict(df.tail(1))[0])


# --------------------------------------------------------------------------- #
# Baseline: tomorrow = today
# --------------------------------------------------------------------------- #
class BaselinePredictor(BasePredictor):
    name = "baseline"

    def fit(self, df: pd.DataFrame) -> "BaselinePredictor":
        return self  # nothing to learn

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["close"].to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# Linear regression
# --------------------------------------------------------------------------- #
class LinearRegressionPredictor(BasePredictor):
    name = "linear_regression"

    def __init__(self, features: Optional[List[str]] = None):
        self.features = features or DEFAULT_FEATURES
        self._model = LinearRegression()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "LinearRegressionPredictor":
        self._check_columns(df, include_target=True)
        data = df[self.features + [TARGET_COL]].dropna()
        self._model.fit(data[self.features], data[TARGET_COL])
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        self._check_columns(df)
        return self._model.predict(df[self.features])

    def _check_columns(self, df: pd.DataFrame, include_target: bool = False) -> None:
        needed = self.features + ([TARGET_COL] if include_target else [])
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
class ModelEvaluator:
    """Splits by time (never shuffle stock data) and reports MAE / RMSE."""

    def __init__(self, test_fraction: float = 0.2):
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1.")
        self.test_fraction = test_fraction

    def time_split(self, df: pd.DataFrame):
        """Oldest rows -> train, newest rows -> test."""
        df = df.dropna(subset=[TARGET_COL])
        cut = int(len(df) * (1 - self.test_fraction))
        return df.iloc[:cut], df.iloc[cut:]

    @staticmethod
    def mae(y_true, y_pred) -> float:
        return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    @staticmethod
    def rmse(y_true, y_pred) -> float:
        return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

    def evaluate(self, model: BasePredictor, df: pd.DataFrame) -> Dict[str, float]:
        train, test = self.time_split(df)
        model.fit(train)
        preds = model.predict(test)
        y = test[TARGET_COL].to_numpy()
        return {"MAE": self.mae(y, preds), "RMSE": self.rmse(y, preds)}

    def compare(self, models: List[BasePredictor], df: pd.DataFrame) -> pd.DataFrame:
        rows = {m.name: self.evaluate(m, df) for m in models}
        return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Quick self-test with made-up data:  python model.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, 300))
    demo = pd.DataFrame({"close": close})
    demo["ma_5"] = demo["close"].rolling(5).mean()
    demo["ma_20"] = demo["close"].rolling(20).mean()
    demo["return_1d"] = demo["close"].pct_change()
    demo[TARGET_COL] = demo["close"].shift(-1)
    demo = demo.dropna(subset=["ma_5", "ma_20", "return_1d"])

    evaluator = ModelEvaluator()
    print(evaluator.compare([BaselinePredictor(), LinearRegressionPredictor()], demo))