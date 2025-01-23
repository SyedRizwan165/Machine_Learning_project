import sys
import pandas as pd
import os
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        """
        Predict the target value(s) for the given input features using the trained model and preprocessor.
        """
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            print("Loading model and preprocessor...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            print("Transforming input features...")
            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)
            
            return predictions

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str,
                 lunch: str, test_preparation_course: str, math_score: int, writing_score: int):
        """
        Initialize the custom data object with user-provided input features.
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.math_score = math_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Convert the input features into a Pandas DataFrame.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "math_score": [self.math_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
