from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os

# Load the trained model
model = joblib.load("xgb_model.pkl")  # Ensure this matches your model file name

# Initialize Flask app
app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if a file is uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # Check if the file is a CSV
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.endswith(".csv"):
            return jsonify({"error": "File must be a CSV"}), 400

        # Save the file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Load the CSV file
        df = pd.read_csv(file_path)

        # Check if the CSV has the required columns
        if "Features" not in df.columns:
            return jsonify({"error": "CSV must have a 'Features' column"}), 400

        # Convert the 'Features' column to a list of lists
        features_list = df["Features"].apply(lambda x: list(map(float, x.split(',')))).tolist()

        # Convert the input features into a NumPy array
        features = np.array(features_list)

        # Make predictions
        predictions = model.predict(features)
        prediction_probas = model.predict_proba(features)

        # Prepare the results (only predictions and confidence scores)
        results = []
        for pred, proba in zip(predictions, prediction_probas):
            # Map prediction to "Unregulated Arousal" or "Regulated Arousal"
            prediction_label = "Regulated Arousal" if pred == 1 else "Unregulated Arousal"
            results.append({
                "Prediction": prediction_label,
                "Confidence_Class_0": float(proba[0]),
                "Confidence_Class_1": float(proba[1])
            })

        # Return the results as JSON
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up: Delete the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
