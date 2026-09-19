import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from app import app
from src.config import DATA_PATH, TARGET
from src.components.data_ingestion import validate_data
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import regression_metrics
from src.pipeline.predict_pipeline import PredictPipeline
from src.utils import evaluate_models, load_object, save_object


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(DATA_PATH)
        cls.X = cls.data.drop(columns=TARGET)
        cls.y = cls.data[TARGET]
        cls.pipeline = Pipeline([
            ("preprocessor", DataTransformation().get_data_transformer_object()),
            ("model", LinearRegression()),
        ]).fit(cls.X.iloc[:800], cls.y.iloc[:800])

    def test_bundled_dataset(self):
        validate_data(self.data)
        self.assertEqual(self.data.shape, (1000, 8))

    def test_reject_corrupt_data(self):
        for bad in [self.data.drop(columns=TARGET),
                    pd.concat([self.data, self.data.iloc[:1]]),
                    self.data.assign(math_score=np.nan),
                    self.data.assign(math_score=101),
                    self.data.assign(gender="invalid")]:
            with self.assertRaises(ValueError):
                validate_data(bad)

    def test_rmse_is_sqrt_mse(self):
        result = regression_metrics([0, 4], [0, 0])
        self.assertAlmostEqual(result["rmse"], np.sqrt(8))
        self.assertEqual(result["mae"], 2)

    def test_cv_fits_preprocessor_per_fold(self):
        # Three fold fits plus one full training refit; no preprocessing of a test set.
        from sklearn.compose import ColumnTransformer
        fitted_sizes = []
        original = ColumnTransformer.fit_transform

        def record(transformer, X, *args, **kwargs):
            fitted_sizes.append(len(X))
            return original(transformer, X, *args, **kwargs)

        with patch.object(ColumnTransformer, "fit_transform", record):
            report, fitted = evaluate_models(self.X.iloc[:90], self.y.iloc[:90],
                {"linear": LinearRegression()}, {"linear": {}})
        self.assertEqual(sorted(fitted_sizes), [60, 60, 60, 90])
        self.assertTrue(np.isfinite(report["linear"]["cv_r2_mean"]))
        self.assertEqual(len(fitted["linear"].predict(self.X.iloc[90:95])), 5)

    def test_selection_uses_cv_not_test_scores(self):
        from src.components.model_trainer import ModelTrainer
        from sklearn.dummy import DummyRegressor
        # Deliberately give the worse test predictor the better CV score.
        poor = DummyRegressor(strategy="constant", constant=-100).fit(self.X, self.y)
        good = self.pipeline
        report = {"cv_winner": {"cv_r2_mean": 0.9}, "test_winner": {"cv_r2_mean": 0.8}}
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            with patch("src.components.model_trainer.evaluate_models",
                       return_value=(report, {"cv_winner": poor, "test_winner": good})), \
                 patch("src.components.model_trainer.ARTIFACTS", artifact_dir), \
                 patch("src.components.model_trainer.MODEL_PATH", artifact_dir / "model.pkl"):
                result = ModelTrainer().initiate_model_trainer(
                    self.data.iloc[:800], self.data.iloc[800:])
            self.assertEqual(result["selected_model"], "cv_winner")
            self.assertLess(result["test"]["r2"], 0)

    def test_serialization_and_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            save_object(path, self.pipeline)
            with patch("src.pipeline.predict_pipeline.MODEL_PATH", path):
                actual = PredictPipeline().predict(self.X.iloc[800:])
            np.testing.assert_allclose(actual, self.pipeline.predict(self.X.iloc[800:]))
            self.assertIsInstance(load_object(path), Pipeline)

    def test_unknown_category_and_missing_numeric(self):
        row = self.X.iloc[:1].copy()
        row["race_ethnicity"] = "new category"
        row["reading_score"] = np.nan
        self.assertTrue(np.isfinite(self.pipeline.predict(row)).all())

    def form(self):
        row = self.X.iloc[0].to_dict()
        row["ethnicity"] = row.pop("race_ethnicity")
        return row

    def test_routes_and_valid_post(self):
        client = app.test_client()
        for route in ["/", "/predictdata"]:
            self.assertEqual(client.get(route).status_code, 200)
        with patch("app.PredictPipeline.predict", side_effect=self.pipeline.predict):
            response = client.post("/predictdata", data=self.form())
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Estimated maths score:", response.data)

    def test_invalid_forms(self):
        for column, value in [("reading_score", ""), ("reading_score", "nan"),
                              ("reading_score", "inf"), ("writing_score", -1),
                              ("writing_score", 101), ("gender", "invalid")]:
            form = self.form()
            form[column] = value
            self.assertEqual(app.test_client().post("/predictdata", data=form).status_code, 400)
        self.assertEqual(app.test_client().post("/predictdata", data={}).status_code, 400)

    def test_missing_model(self):
        with patch("app.PredictPipeline.predict", side_effect=FileNotFoundError):
            response = app.test_client().post("/predictdata", data=self.form())
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
