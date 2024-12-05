from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('graph_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['graph']
    graph_data = np.array(data).reshape(1, -1)  # Flatten the array
    prediction = model.predict(graph_data)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)