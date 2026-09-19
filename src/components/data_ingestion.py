"""Validate the bundled data and create a deterministic train/test split."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import ARTIFACTS, DATA_PATH, TARGET, NUMERIC, CATEGORIES


def validate_data(data):
    expected = set(NUMERIC) | set(CATEGORIES) | {TARGET}
    if set(data.columns) != expected:
        raise ValueError("Dataset columns do not match the expected schema")
    if data.empty or data.isna().any().any() or data.duplicated().any():
        raise ValueError("Dataset must be nonempty with no missing or duplicate rows")
    for column in NUMERIC + [TARGET]:
        values = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(values).all() or not values.between(0, 100).all():
            raise ValueError(f"{column} must contain finite scores between 0 and 100")
    for column, allowed in CATEGORIES.items():
        if not data[column].isin(allowed).all():
            raise ValueError(f"Unexpected category in {column}")


class DataIngestion:
    def initiate_data_ingestion(self):
        data = pd.read_csv(DATA_PATH)
        validate_data(data)
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        train_path, test_path = ARTIFACTS / "train.csv", ARTIFACTS / "test.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        return train_path, test_path


if __name__ == "__main__":
    from src.components.model_trainer import ModelTrainer
    train_path, test_path = DataIngestion().initiate_data_ingestion()
    print(ModelTrainer().initiate_model_trainer(
        pd.read_csv(train_path), pd.read_csv(test_path)))
