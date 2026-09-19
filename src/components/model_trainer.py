"""Select by training CV, then evaluate the selected pipeline on held-out data."""
import hashlib
import json
import platform
from importlib.metadata import version
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.config import ARTIFACTS, DATA_PATH, MODEL_PATH, TARGET
from src.utils import evaluate_models, save_object


def regression_metrics(actual, predicted):
    return {"r2": float(r2_score(actual, predicted)),
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted)))}


class ModelTrainer:
    def initiate_model_trainer(self, train, test):
        X_train, y_train = train.drop(columns=TARGET), train[TARGET]
        X_test, y_test = test.drop(columns=TARGET), test[TARGET]
        models = {
            "Random Forest": RandomForestRegressor(random_state=42, n_jobs=1),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Linear Regression": LinearRegression(),
            "XGBRegressor": XGBRegressor(random_state=42, n_jobs=1),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False, random_seed=42,
                                                       thread_count=1, allow_writing_files=False),
            "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
        }
        params = {'Decision Tree': {'criterion': ['squared_error',
                                         'absolute_error',
                                         'poisson']},
         'Random Forest': {'n_estimators': [8, 16, 32, 64, 128, 256]},
         'Gradient Boosting': {'learning_rate': [0.1, 0.01, 0.05, 0.001],
                               'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                               'n_estimators': [8, 16, 32, 64, 128, 256]},
         'Linear Regression': {},
         'XGBRegressor': {'learning_rate': [0.1, 0.01, 0.05, 0.001],
                          'n_estimators': [8, 16, 32, 64, 128, 256]},
         'CatBoosting Regressor': {'depth': [6, 8, 10],
                                   'learning_rate': [0.01, 0.05, 0.1],
                                   'iterations': [30, 50, 100]},
         'AdaBoost Regressor': {'learning_rate': [0.1, 0.01, 0.5, 0.001],
                                'n_estimators': [8, 16, 32, 64, 128, 256]}}
        report, fitted = evaluate_models(X_train, y_train, models, params)
        best_name = max(report, key=lambda name: report[name]["cv_r2_mean"])
        best = fitted[best_name]
        baseline = DummyRegressor(strategy="mean").fit(np.zeros((len(train), 1)), y_train)
        result = {
            "selected_model": best_name,
            "selection": "highest mean training-only 3-fold shuffled CV R2; seed 42",
            "train_rows": len(train), "test_rows": len(test), "split_seed": 42,
            "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "versions": {name: version(name) for name in
                         ["numpy", "pandas", "scikit-learn", "catboost", "xgboost"]},
            "cross_validation": report,
            "test": regression_metrics(y_test, best.predict(X_test)),
            "dummy_test": regression_metrics(y_test, baseline.predict(np.zeros((len(test), 1)))),
            "caveat": "This split was inspected in the original project; it is not a new external validation set.",
        }
        save_object(MODEL_PATH, best)
        (ARTIFACTS / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
