import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

# Import custom modules for exception handling, logging, and components
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_training import ModelTrainingConfig
from src.components.model_training import ModelTraining

# Configuration class for Data Ingestion paths
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')  # Path to save the training data
    test_data_path: str = os.path.join('artifacts', 'test.csv')   # Path to save the testing data
    raw_data_path: str = os.path.join('artifacts', 'data.csv')    # Path to save the raw data

# Data Ingestion class to handle the ingestion process
class DataIngestion:
    def __init__(self):
        # Initialize the ingestion configuration
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Method to ingest data: 
        1. Load raw data from a CSV file.
        2. Split the data into training and testing datasets.
        3. Save the datasets to the specified paths.
        """
        logging.info("Entered the data ingestion method/component")
        try:
            # Load the dataset into a Pandas DataFrame
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Dataset loaded successfully into a DataFrame')

            # Create the directory structure for saving the data if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Perform train-test split
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test datasets to their respective paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion process completed successfully")

            # Return the paths to the train and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # Handle exceptions and log the error
            raise CustomException(e, sys)

# Main entry point for the script
if __name__ == "__main__":
    # Step 1: Initialize and execute the Data Ingestion process
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Step 2: Perform Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Step 3: Train the model using the transformed data
    modeltrainer = ModelTraining()
    print(modeltrainer.initiate_model_training(train_arr, test_arr))


