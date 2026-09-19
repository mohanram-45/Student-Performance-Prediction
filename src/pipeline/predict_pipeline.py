import numpy as np
import pandas as pd
from src.config import MODEL_PATH, CATEGORIES, NUMERIC
from src.utils import load_object


class PredictPipeline:
    def predict(self, features):
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Run python -m src.components.data_ingestion first")
        pipeline = load_object(MODEL_PATH)
        return pipeline.predict(features)


class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            data = pd.DataFrame(custom_data_input_dict)
            for column, allowed in CATEGORIES.items():
                if not data[column].isin(allowed).all():
                    raise ValueError(f"Select a valid value for {column}")
            for column in NUMERIC:
                data[column] = pd.to_numeric(data[column], errors="raise")
                if not np.isfinite(data[column]).all() or not data[column].between(0, 100).all():
                    raise ValueError(f"{column} must be between 0 and 100")
            return data

        except (TypeError, ValueError) as error:
            raise ValueError(str(error)) from error