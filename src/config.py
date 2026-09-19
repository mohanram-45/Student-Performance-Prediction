"""Paths shared by training and inference, independent of the working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "notebook" / "data" / "stud.csv"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model.pkl"
TARGET = "math_score"
NUMERIC = ["writing_score", "reading_score"]
CATEGORIES = {
    "gender": ["female", "male"],
    "race_ethnicity": [f"group {letter}" for letter in "ABCDE"],
    "parental_level_of_education": ["associate's degree", "bachelor's degree",
        "high school", "master's degree", "some college", "some high school"],
    "lunch": ["free/reduced", "standard"],
    "test_preparation_course": ["none", "completed"],
}
