from flask import Flask, request, jsonify
import numpy as np
import joblib

# Load the trained model
model = joblib.load("xgb_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Log the request headers and body
        app.logger.info("Headers: %s", request.headers)
        app.logger.info("Body: %s", request.get_data())

        # Check the Content-Type header
        if request.headers.get("Content-Type") != "application/json":
            return jsonify({"error": "Content-Type must be application/json"}), 415

        # Get the input data from the request
        data = request.json

        # Check if 'features' key exists
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        # Convert the input features into a NumPy array
        features = np.array(data["features"]).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)

        # Return the prediction as a JSON response
        return jsonify({
            "prediction": int(prediction[0]),
            "probabilities": prediction_proba.tolist()
        })

    except Exception as e:
        # Return a 500 error with the exception message
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Random Forest Model API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("xgb_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the request
    data = request.json

    # Convert the input features into a NumPy array
    features = np.array(data["features"]).reshape(1, -1)  # Reshape to (1, n_features)

    # Make a prediction
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)

    # Return the prediction as a JSON response
    return jsonify({
        "prediction": int(prediction[0]),  # Convert numpy int64 to Python int
        "probabilities": prediction_proba.tolist()  # Convert numpy array to list
    })

@app.route("/")
def home():
    return "Model API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
