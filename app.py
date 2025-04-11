from flask import Flask, request, jsonify
from utils.mental_health_predict import predict_mental_health
import traceback

app = Flask(__name__)

@app.route('/predict/mental-health', methods=['POST'])
def mental_health():
    try:
        data = request.get_json()

        # Validate input
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        prediction = predict_mental_health(data)
        return jsonify({'prediction': prediction})

    except Exception as e:
        # Log and return error message
        print("Error occurred:\n", traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    
    
# output 0 --> healthy ,  1 --> moderate , 2 --> At Risk 
if __name__ == '__main__':
    app.run(debug=True)
