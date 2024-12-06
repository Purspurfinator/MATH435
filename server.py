from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model
model = joblib.load('graph_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logging.debug("Received request: %s", request.json)
    data = request.json['graph']
    graph_data = np.array(data).reshape(1, -1)  # Flatten the array
    logging.debug("Graph data: %s", graph_data)
    prediction = model.predict(graph_data)
    logging.debug("Prediction: %s", prediction[0])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)