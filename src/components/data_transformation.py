import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

# Configuration class for Data Transformation
@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')  # Path to save the preprocessor object

# Data Transformation class for handling data preprocessing and transformation
class DataTransformation:
    def __init__(self):
        # Initialize the configuration for Data Transformation
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a ColumnTransformer object for data preprocessing.
        Includes pipelines for both numerical and categorical features.
        """
        try:
            # Define numerical and categorical columns
            numerical_columns = ['math_score', 'writing_score']
            categorical_columns = [
                "gender",
                "lunch",
                "race_ethnicity",
                "parental_level_of_education",
                "test_preparation_course",
            ]

            # Numerical pipeline: handles missing values and scales the data
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")

            # Categorical pipeline: handles missing values, encodes categorical data, and scales it
            cat_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")

            # Combine numerical and categorical pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            # Handle exceptions and log the error
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies data transformation on training and testing datasets.
        Returns transformed train and test arrays and saves the preprocessor object.
        """
        try:
            # Load training and testing datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data successfully loaded")

            # Obtain the preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Define target column and numerical columns
            target_column_name = "reading_score"
            numerical_columns = ['math_score', 'writing_score']

            # Separate input features and target variable for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target variable for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training and testing dataframes")

            # Apply preprocessing to training and testing input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine input features and target variable for training and testing datasets
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saving preprocessing object")

            # Save the preprocessor object to a file
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )

            # Return transformed train and test arrays and the preprocessor file path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )

        except Exception as e:
            # Handle exceptions and log the error
            raise CustomException(e, sys)

            
            
