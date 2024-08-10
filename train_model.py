import joblib
import numpy as np

# Load the model from the .sav file
file_path = 'symptoms.sav'
model = joblib.load(file_path)


input_data = np.array([
    [12,4, 28, 21.85, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],  # Example set of features

])

# Use the model to make predictions
predictions = model.predict(input_data)

# Print the predictions
print("Predictions:", predictions)