# Stock Prediction App (Student Project)

## Who owns what
| Person | File | Classes |
|---|---|---|
| 1 — Data | `data_handler.py` | `StockQuery`, `DataFetcher`, `DataCleaner`, `FeatureBuilder` |
| 2 — Model | `model.py` | `PredictionResult`, `BasePredictor`, `BaselinePredictor`, `LinearRegressionPredictor`, `ModelEvaluator` |
| 3 — App/UI | `app.py` | `InputHandler`, `Visualizer`, `StockPredictionApp` |

## Data flow
```
InputHandler -> StockQuery -> DataFetcher -> DataCleaner -> FeatureBuilder
     -> ModelEvaluator(predictor) -> PredictionResult -> Visualizer
```

## Agree on these first (the "contracts")
- DataFrame columns: `Date, Open, High, Low, Close, Volume, MA_5, Return, Target`
- `PredictionResult` fields (in `model.py`)
- Feature column list used in `StockPredictionApp.choose_model`

## Setup
```
pip install pandas yfinance scikit-learn matplotlib
python app.py
```

## Tips
- Person 2 can start before Person 1 is done by using a small fake DataFrame.
- Person 3 can test charts with a hand-made `PredictionResult`.
- Split train/test by time, never shuffle.
- Always compare against `BaselinePredictor`; beating "tomorrow = today" is harder than it sounds.
