import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
import pickle
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException  



# Function to save an object to a file
def save_object(file_path, obj):
    try:
        # Create the directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the specified file path
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception in case of an error
        raise CustomException(e, sys)
    

# Function to evaluate different models using GridSearchCV
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}  # Dictionary to store evaluation results

        # Iterate through each model in the provided models dictionary
        for i in range(len(list(models))):
            model = list(models.values())[i]  # Get model instance
            para = param[list(models.keys())[i]]  # Get parameters for the model

            # Perform GridSearchCV to find the best parameters
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Set the best parameters to the model
            model.set_params(**gs.best_params_)

            # Train the model with the best parameters
            model.fit(X_train, y_train)

            # Predict on train and test data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 score for training and testing
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test R2 score in the report
            report[list(models.keys())[i]] = test_model_score

            return report  # Return the report after evaluating all models

    except Exception as e:
        # Raise a custom exception in case of an error
        raise CustomException(e, sys)
    

# Function to load an object from a file
def load_object(file_path):
    try:
        # Load the object from the specified file path
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        # Raise a custom exception in case of an error
        raise CustomException(e, sys)
