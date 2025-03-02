import requests
import pandas as pd

# API endpoint
url = "http://127.0.0.1:5000/predict"

# Load the CSV file
file_path = "filtered_test_data.csv"
df = pd.read_csv(file_path)

# Check if the CSV has the required columns
if "Features" not in df.columns:
    raise ValueError("CSV must have a 'Features' column")

# Convert the 'Features' column to a list of lists
features_list = df["Features"].apply(lambda x: list(map(float, x.split(',')))).tolist()

# Prepare the JSON payload
data = {
    "features": features_list  # Send all rows at once
}

# Set the Content-Type header to application/json
headers = {"Content-Type": "application/json"}

# Send the POST request
response = requests.post(url, json=data, headers=headers)

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
