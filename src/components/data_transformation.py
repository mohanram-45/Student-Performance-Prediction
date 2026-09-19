"""Preprocessing fitted inside each training cross-validation fold."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import NUMERIC, CATEGORIES


class DataTransformation:
    def get_data_transformer_object(self):
        numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler())])
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ("scaler", StandardScaler(with_mean=False)),
        ])
        return ColumnTransformer([
            ("numeric", numeric, NUMERIC),
            ("categorical", categorical, list(CATEGORIES)),
        ])
