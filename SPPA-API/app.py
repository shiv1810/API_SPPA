# give me a simple layout of a flask api that runs hello world?
# app.py
from flask import Flask, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the SVM model
svm_model = joblib.load('svm_model.pkl')

# Sample input values (replace with your actual input values)
sample_input_values = {
    "age": 15,
    "Medu": 4,
    "studytime": 5,
    "failures": 0,
    "goout": 2,
    "health": 5,
    "absences": 8,
    "G1": 15,
    "G2": 14,
    "sex_enc": 0,
    "higher_enc": 1,
}

@app.route("/")
def display_prediction():
    # Convert the sample input values to a DataFrame
    input_data = pd.DataFrame([sample_input_values])

    # Perform prediction
    prediction = svm_model.predict(input_data)

    # Convert the prediction to a dictionary
    result = {'prediction': prediction.tolist()}

    return jsonify(result)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
