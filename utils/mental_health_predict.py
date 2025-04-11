import joblib
import numpy as np

# Load the trained model
model = joblib.load('models/mental_health_model.pkl')

def predict_mental_health(data):
    expected_features = [
        'age',
        'sleep_hours',
        'stress_level',
        'work_life_balance',
        'physical_activity',
        'social_support',
        'chronic_illness',
        'anxious_days',
        'depressed_days'
    ]

    try:
        input_features = [data[feature] for feature in expected_features]
    except KeyError as e:
        raise ValueError(f"Missing input field: {e.args[0]}")

    features = np.array([input_features])
    prediction = model.predict(features)[0]

    # Convert numpy data type to native Python type
    return str(prediction)  # or use int(prediction) if it's a number label
