import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Load dataset for frequency encoding
data = pd.read_csv('./final.csv')

# Load the trained recommendation model
model_path = "recommendation_model.pkl"
try:
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        label_encoders = model_data['label_encoders']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model, label_encoders, scaler, feature_cols = None, None, None, None

# Compute frequency encodings once
user_counts = data['user_id'].value_counts()
product_counts = data['product_id'].value_counts()

target_columns = ['stationary', 'electronics', 'apparel', 'other']

# Preprocess input data


def preprocess_input(data):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Encode categorical features safely
        for col in ['brand', 'cat_0']:
            if col in df.columns and col in label_encoders:
                df[col] = df[col].map(lambda x: label_encoders[col].transform([x])[
                                      0] if x in label_encoders[col].classes_ else -1)

        # Apply frequency encoding
        df['user_id_freq'] = df['user_id'].map(
            user_counts) / len(data) if 'user_id' in df.columns else 0
        df['product_id_freq'] = df['product_id'].map(
            product_counts) / len(data) if 'product_id' in df.columns else 0

        # Scale numerical features
        if 'price' in df.columns:
            df[['price']] = scaler.transform(df[['price']])

        return df[feature_cols]
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

# API Endpoint for Prediction


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        preprocessed_data = preprocess_input(data)

        if preprocessed_data is None:
            return jsonify({"error": "Invalid input data"}), 400

        # Make prediction
        prediction = model.predict(preprocessed_data)
        probabilities = model.predict_proba(preprocessed_data)

        # Extract probability of each category
        class_probs = {target_columns[i]: probabilities[0][i]
                       for i in range(len(target_columns))}

        # Get the recommended category
        recommended_category = target_columns[np.argmax(
            list(class_probs.values()))]

        return jsonify({
            "predicted_categories": prediction.tolist(),
            "probabilities": class_probs,
            "recommended_category": recommended_category
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Failed to process request"}), 500


# Start the server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
