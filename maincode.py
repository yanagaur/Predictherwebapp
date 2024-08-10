from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# Load the model from the .sav file
file_path = 'symptoms.sav'
model = joblib.load(file_path)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and domains

# Define the BMI insights function
def BMI_Insights(x):
    if x < 18:
        return 'A'
    elif x >= 18 and x < 26:
        return 'B'
    else:
        return 'C'

@app.route('/noreport', methods=['POST'])
def noreport():
    try:
        # Log the content type
        content_type = request.content_type
        print("Content-Type:", content_type)

        # Ensure the content type is application/json
        if 'application/json' not in content_type:
            return jsonify({'error': '415 Unsupported Media Type: Content-Type must be application/json'}), 415

        # Get the data from the POST request
        data = request.json

        # Log the received data
        print("Received data:", data)

        # Validate and extract input data
        try:
            age = int(data['age'])
            cycle_length = int(data['cycleLength']) if data['cycleLength'] else 0  # Default to 0 if empty
            waist = float(data['waist']) if data['waist'] else 0.0  # Default to 0.0 if empty

            features = [
                age,
                int(data['Cycle(R/I)'].lower() == 'regular'),
                cycle_length,
                waist,
                int(data['weightGain'] == 'Y'),
                int(data['hairGrowth'] == 'Y'),
                int(data['skinDarkening'] == 'Y'),
                int(data['hairLoss'] == 'Y'),
                int(data['pimples'] == 'Y'),
                float(data['fastFood'] == 'Y'),
                int(data['regExercise'] == 'Y')
            ]

            # Calculate BMI category
            bmi_value = float(data['bmi']) if data['bmi'] else 0.0  # Default to 0.0 if empty
            bmi_category = BMI_Insights(bmi_value)

            # Initialize BMI category variables
            bmi_a = 1.0 if bmi_category == 'A' else 0.0
            bmi_b = 1.0 if bmi_category == 'B' else 0.0
            bmi_c = 1.0 if bmi_category == 'C' else 0.0

            # Append BMI category variables to the features list
            features.extend([bmi_a, bmi_b, bmi_c])

        except (KeyError, ValueError) as e:
            return jsonify({'error': f'Invalid input data: {str(e)}'}), 400

        # Convert to a 2D numpy array
        features = np.array(features, dtype=float).reshape(1, -1)

        # Use the model to make predictions
        predictions = model.predict(features)

        # Return the predictions and other results as a JSON response
        return jsonify({
            'predictions': predictions.tolist(),
            'bmi_a': bmi_a,
            'bmi_b': bmi_b,
            'bmi_c': bmi_c
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Vercel serverless function handler
def handler(request, response):
    return app(request, response)
