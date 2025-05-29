from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model and feature order
model = joblib.load("rf_model.pkl")

with open("feature_order.txt") as f:
    feature_order = [line.strip() for line in f.readlines()]

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate input
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' key in JSON"}), 400

        feature_vector = data["features"]

        if len(feature_vector) != len(feature_order):
            return jsonify({"error": f"Expected {len(feature_order)} features, got {len(feature_vector)}"}), 400

        # Convert to 2D array for prediction
        X = np.array(feature_vector).reshape(1, -1)

        # Make prediction
        prediction = model.predict(X)[0]
        result = "phishing" if prediction == 0 else "legitimate"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

