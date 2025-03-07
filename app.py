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


def predict_category(sample_data):
    try:
        # Make a copy to avoid modifying the original
        sample = sample_data.copy()

        # Apply the same preprocessing steps
        for col in ['brand', 'cat_0']:
            if col in sample.columns and col in label_encoders:
                # Handle unseen labels by setting them to -1
                sample[col] = sample[col].apply(lambda x: label_encoders[col].transform([x])[
                                                0] if x in label_encoders[col].classes_ else -1)

        # Apply frequency encoding
        if 'user_id' in sample.columns:
            sample['user_id_freq'] = sample['user_id'].map(
                user_counts) / len(data)

        if 'product_id' in sample.columns:
            sample['product_id_freq'] = sample['product_id'].map(
                product_counts) / len(data)

        # Scale numerical features
        if 'price' in sample.columns:
            sample[['price']] = scaler.transform(sample[['price']])

        # Make prediction
        prediction = model.predict(sample[feature_cols])
        probabilities = model.predict_proba(sample[feature_cols])

        # Create a DataFrame for the binary predictions
        binary_predictions = pd.DataFrame(prediction, columns=target_columns)

        # Calculate probabilities for each class
        class_probs = {}
        for i, target in enumerate(target_columns):
            class_probs[target] = probabilities[i][0][1]

        # Get the recommended category
        recommended_category = target_columns[np.argmax(
            list(class_probs.values()))]

        return binary_predictions, class_probs, recommended_category
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

# API Endpoint for Prediction


@app.route("/predict", methods=["POST"])
def predict():
    try:
        request_data = request.json
        sample_data = pd.DataFrame([request_data])

        print("Sample input:")
        print(sample_data)

        binary_predictions, class_probs, recommended_category = predict_category(
            sample_data)

        if binary_predictions is None:
            return jsonify({"error": "Invalid input data"}), 400

        print("\nPredicted categories (binary):")
        print(binary_predictions)

        print("\nPredicted probabilities for each category:")
        for category, prob in class_probs.items():
            print(f"{category}: {prob:.4f}")

        print("\nRecommended category:", recommended_category)

        return jsonify({
            "predicted_categories": binary_predictions.to_dict(orient='records'),
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
