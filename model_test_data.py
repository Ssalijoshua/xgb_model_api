import pandas as pd


# Load the CSV file
data = pd.read_csv('test_windowed.csv')

# Filter out rows where the subject is not 18 years old
filtered_data = data[data['Subject'] == 18]

# Drop the target column (assuming it's named 'target')
filtered_data = filtered_data.drop(columns=['Label'], errors='ignore')

# Get the 5th row (index 4 because Python is 0-based)
test = filtered_data.iloc[4]

# Display the row
print(test)

# Save as CSV (but single rows are usually saved as Series, so converting to DataFrame first)
test.to_frame().T.to_csv('filtered_test_data.csv', index=False)
