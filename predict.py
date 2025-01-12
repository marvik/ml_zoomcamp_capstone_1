import pickle
import pandas as pd
from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np

# Load the model and DictVectorizer
model_file = 'xgboost_model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('online_shopping')

@app.route('/predict', methods=['POST'])
def predict():
    customer_data = request.get_json()

    # Convert to DataFrame
    df = pd.DataFrame([customer_data])

    # Convert boolean columns to numerical (True -> 1, False -> 0)
    bool_columns = df.select_dtypes(include='bool').columns
    for col in bool_columns:
        df[col] = df[col].astype(int)

    # Transform data using DictVectorizer
    customer_dict = df.to_dict(orient='records')
    X = dv.transform(customer_dict)

    # Convert to DMatrix
    features = dv.get_feature_names_out().tolist()
    dtest = xgb.DMatrix(X, feature_names=features)

    # Make prediction
    y_pred = model.predict(dtest)[0]  # Get the probability

    # Convert to boolean prediction (True if probability >= 0.5)
    revenue = y_pred >= 0.5  # Correct variable name to 'revenue'

    result = {
        'revenue_probability': float(y_pred),
        'revenue': bool(revenue)  # Use the correct variable 'revenue' here
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)