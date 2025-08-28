from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import logging

# Load model and feature order
model = joblib.load("rf_model.pkl")
with open("feature_order.txt") as f:
    feature_order = [line.strip() for line in f.readlines()]

# Configure Flask
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate input
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' key in JSON"}), 400

        input_features = data["features"]

        # If features are passed as a dictionary
        if isinstance(input_features, dict):
            feature_vector = [input_features.get(feat, 0) for feat in feature_order]
        elif isinstance(input_features, list):
            if len(input_features) != len(feature_order):
                return jsonify({
                    "error": f"Expected {len(feature_order)} features, got {len(input_features)}"
                }), 400
            feature_vector = input_features
        else:
            return jsonify({"error": "'features' must be a list or dict"}), 400

        # Reshape for prediction
        X = np.array(feature_vector).reshape(1, -1)

        # Predict
        prediction = model.predict(X)[0]
        confidence = float(np.max(model.predict_proba(X)))
        result = "phishing" if prediction == 0 else "legitimate"

        # Logging
        logging.info(f"Input: {feature_vector}")
        logging.info(f"Prediction: {result} ({confidence:.4f})")

        return jsonify({
            "prediction": result,
            "confidence": round(confidence, 4),
            "features_checked": len(feature_vector)
        })

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)
