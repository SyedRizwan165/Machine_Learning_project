from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create the Flask application
application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html') 

# Route for handling data prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # Render the data input form
        return render_template('home.html')
    else:
        # Collect data from the form
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            math_score=float(request.form.get('math_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        
        # Convert the data into a DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        # Initialize the prediction pipeline
        print("Before Prediction")
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        
        # Make predictions
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        
        # Render the results on the home page
        return render_template('home.html', results=results[0])

# Run the application
if __name__ == "__main__":
    app.run(host="0.0.0.0")