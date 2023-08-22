from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model (replace with your actual model loading code)
model = joblib.load("trained_model.joblib")

# Define an API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get input data from the POST request
    features = data['features']  # Extract features from the input data
    prediction = model.predict([features])[0]  # Make a prediction using the loaded model
    return jsonify({'prediction': prediction})  # Return prediction as JSON response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Run the Flask app on host 0.0.0.0 (accessible from any IP) and port 5000
