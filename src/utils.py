"""Persistence and training-only model selection."""
import pickle
from pathlib import Path
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from src.components.data_transformation import DataTransformation


def save_object(file_path, obj):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(obj, stream)


def load_object(file_path):
    # Only load artifacts produced locally from trusted code.
    with Path(file_path).open("rb") as stream:
        return pickle.load(stream)


def evaluate_models(X_train, y_train, models, param):
    report, fitted = {}, {}
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", DataTransformation().get_data_transformer_object()),
            ("model", model),
        ])
        search = GridSearchCV(pipeline,
            {f"model__{key}": values for key, values in param[name].items()},
            cv=cv, scoring="r2", n_jobs=1, error_score="raise")
        search.fit(X_train, y_train)
        report[name] = {"cv_r2_mean": float(search.best_score_),
            "cv_r2_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "best_params": search.best_params_}
        fitted[name] = search.best_estimator_
        print(f"{name}: CV R2={search.best_score_:.4f}", flush=True)
    return report, fitted
