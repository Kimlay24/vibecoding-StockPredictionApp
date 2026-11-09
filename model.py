"""
PERSON 2 — MODEL
Owns: training, predicting, and measuring accuracy.
Receives: feature DataFrame from Person 1 (data_handler.py).
Hands off: PredictionResult to Person 3 (app.py).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class PredictionResult:
    """What the model returns. Person 3 displays this."""
    ticker: str
    predicted_prices: list[float]
    dates: list[str]
    metrics: dict = field(default_factory=dict)  # e.g. {"MAE": 1.2, "RMSE": 1.8}


class BasePredictor(ABC):
    """Every model follows this interface so they can be swapped easily."""

    def __init__(self, feature_cols: list[str], target_col: str = "Target"):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.is_trained = False

    @abstractmethod
    def train(self, df: pd.DataFrame) -> None:
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> list[float]:
        ...


class BaselinePredictor(BasePredictor):
    """Simplest model: tomorrow = today. Everything else must beat this."""

    def train(self, df: pd.DataFrame) -> None:
        self.is_trained = True

    def predict(self, df: pd.DataFrame) -> list[float]:
        """TODO: return df['Close'] as the prediction."""
        raise NotImplementedError


class LinearRegressionPredictor(BasePredictor):
    """Main model using scikit-learn LinearRegression."""

    def __init__(self, feature_cols: list[str], target_col: str = "Target"):
        super().__init__(feature_cols, target_col)
        self.model = None  # TODO: sklearn.linear_model.LinearRegression()

    def train(self, df: pd.DataFrame) -> None:
        """TODO: fit self.model on df[feature_cols] -> df[target_col]."""
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> list[float]:
        raise NotImplementedError


class ModelEvaluator:
    """Splits data and scores models."""

    def train_test_split(self, df: pd.DataFrame, test_ratio: float = 0.2):
        """
        TODO: split by TIME (first 80% train, last 20% test).
        Do NOT shuffle — that leaks future data into training.
        """
        raise NotImplementedError

    def evaluate(self, actual: list[float], predicted: list[float]) -> dict:
        """TODO: return {'MAE': ..., 'RMSE': ...}."""
        raise NotImplementedError

    def run(self, predictor: BasePredictor, df: pd.DataFrame, ticker: str) -> PredictionResult:
        """Split, train, predict, score, and package the result."""
        raise NotImplementedError
