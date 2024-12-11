from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS
import logging
import base64
from PIL import Image
import io
import imageio.v2 as imageio
from scipy.signal import find_peaks

app = Flask(__name__)

# Update CORS configuration to allow requests from your GitHub Pages domain
CORS(app, resources={r"/*": {"origins": "https://purspurfinator.github.io"}})

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model and scaler
model = joblib.load('advanced_graph_model.pkl')
scaler = joblib.load('scaler.pkl')

def image_to_matrix(image_path, new_size=(250, 250)):
    img = imageio.imread(image_path)
    img_rgba = Image.fromarray(img).convert('RGBA').resize(new_size)
    
    # Create a white background image
    white_bg = Image.new('RGBA', img_rgba.size, (255, 255, 255, 255))
    
    # Composite the image with the white background
    img_composite = Image.alpha_composite(white_bg, img_rgba).convert('L')
    
    # Convert to binary matrix: path (black) as 1, background (white) as 0
    binary_matrix = (np.array(img_composite) < 128).astype(int)
    
    return binary_matrix

def extract_features_from_matrix(matrix):
    y_values = matrix.mean(axis=1)  # Simplified example, adjust as needed
    x_values = np.linspace(0, len(y_values) - 1, len(y_values))
    
    feature_dict = {}
    
    # Calculate slope
    slopes = np.gradient(y_values, x_values)
    feature_dict['mean_slope'] = np.mean(slopes)
    feature_dict['std_slope'] = np.std(slopes)
    
    # Slopes at the beginning and end of the plot
    feature_dict['start_slope'] = slopes[0]
    feature_dict['end_slope'] = slopes[-1]
    
    # Find peaks (local maxima)
    peaks, _ = find_peaks(y_values)
    feature_dict['num_peaks'] = len(peaks)
    
    # Find valleys (local minima)
    valleys, _ = find_peaks(-y_values)
    feature_dict['num_valleys'] = len(valleys)
    
    # Exponential growth/decay
    if np.all(y_values > 0):
        feature_dict['exp_growth_rate'] = np.mean(np.diff(np.log(y_values)))
    else:
        feature_dict['exp_growth_rate'] = 0
    
    # Specific logic for different types of plots
    feature_dict['is_parabola'] = int(np.all(np.diff(np.sign(np.gradient(slopes))) != 0))  # Quadratic plot
    feature_dict['end_behavior'] = np.sign(y_values[0]) * np.sign(y_values[-1])  # Odd degree polynomial
    feature_dict['even_end_behavior'] = int(np.sign(y_values[0]) == np.sign(y_values[-1]))  # Even degree polynomial
    
    # Maximum number of critical points for polynomials
    feature_dict['num_critical_points'] = len(peaks) + len(valleys)
    
    # Absolute value function behavior
    if len(peaks) == 1 and len(valleys) == 0:
        feature_dict['abs_slope_diff'] = abs(slopes[0] - slopes[-1])
    else:
        feature_dict['abs_slope_diff'] = 0
    
    return np.array(list(feature_dict.values()))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logging.debug("Received request: %s", request.json)
    data = request.json['image']
    
    # Decode the base64 string
    image_data = base64.b64decode(data.split(',')[1])
    image = Image.open(io.BytesIO(image_data))
    image_path = 'user_graph.png'
    image.save(image_path)
    
    # Convert the image to a binary matrix
    graph_matrix = image_to_matrix(image_path)
    
    # Save the binary matrix as a text file for inspection
    np.savetxt('binary_matrix.txt', graph_matrix, fmt='%d')
    print("Saved binary matrix as 'binary_matrix.txt'")
    
    # Verify the shape of the binary matrix
    print(f"Binary matrix shape: {graph_matrix.shape}")
    
    # Extract features from the binary matrix
    features = extract_features_from_matrix(graph_matrix)
    
    # Normalize the features
    features = scaler.transform([features])
    
    # Verify the shape of the normalized array
    print(f"Normalized array shape: {features.shape}")
    
    # Debugging prints before feeding the input to the model
    logging.debug("Graph data shape: %s", features.shape)
    logging.debug("Number of features: %d", features.size)
    
    prediction = model.predict(features)
    logging.debug("Prediction: %s", prediction[0])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)