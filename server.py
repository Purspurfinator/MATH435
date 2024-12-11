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

# Load the trained model
model = joblib.load('advanced_graph_model.pkl')

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
    y_coords, x_coords = np.where(matrix == 1)
    
    # Check if there are any points with value 1
    if len(y_coords) == 0 or len(x_coords) == 0:
        # If no points with value 1, set all features to 0
        feature_dict = {key: 0 for key in [
            'num_peaks', 'num_valleys', 'num_critical_points', 'max_value', 'min_value',
            'width', 'height', 'area', 'symmetry', 'exp_growth_rate', 'is_parabola',
            'end_behavior', 'even_end_behavior', 'is_abs', 'is_even', 'is_odd', 'is_sine'
        ]}
        return np.array(list(feature_dict.values()))
    
    # Sort coordinates by x values
    sorted_indices = np.argsort(x_coords)
    x_values = x_coords[sorted_indices]
    y_values = y_coords[sorted_indices]
    
    # Smooth the y values to reduce noise
    if len(y_values) > 11:  # Ensure there are enough points to apply the filter
        y_values_smooth = savgol_filter(y_values, window_length=11, polyorder=2)
    else:
        y_values_smooth = y_values
    
    feature_dict = {}
    
    # Find peaks (local maxima)
    peaks, _ = find_peaks(y_values_smooth)
    feature_dict['num_peaks'] = len(peaks)
    
    # Find valleys (local minima)
    valleys, _ = find_peaks(-y_values_smooth)
    feature_dict['num_valleys'] = len(valleys)
    
    # Total number of critical points (peaks + valleys)
    feature_dict['num_critical_points'] = len(peaks) + len(valleys)
    
    # Max and Min Values
    feature_dict['max_value'] = np.max(y_values)
    feature_dict['min_value'] = np.min(y_values)
    
    # Width and Height
    feature_dict['width'] = len(x_values)
    feature_dict['height'] = feature_dict['max_value'] - feature_dict['min_value']
    
    # Area under the curve
    feature_dict['area'] = np.sum(y_values)
    
    # Symmetry
    feature_dict['symmetry'] = np.sum(np.abs(y_values - y_values[::-1])) / len(y_values)
    
    # Exponential growth/decay
    if np.all(y_values > 0):
        feature_dict['exp_growth_rate'] = np.mean(np.diff(np.log(y_values)))
    else:
        feature_dict['exp_growth_rate'] = 0
    
    # Specific logic for different types of plots
    feature_dict['is_parabola'] = int(len(peaks) == 1 and len(valleys) == 1)  # Quadratic plot
    
    # End behavior based on edge values
    feature_dict['end_behavior'] = np.sign(y_values[0]) * np.sign(y_values[-1])  # Odd degree polynomial
    feature_dict['even_end_behavior'] = int(np.sign(y_values[0]) == np.sign(y_values[-1]))  # Even degree polynomial
    
    # Absolute value function behavior
    if len(peaks) == 1 and len(valleys) == 0:
        peak_index = peaks[0]
        if peak_index > 0 and peak_index < len(y_values):
            left_segment = y_values[:peak_index]
            right_segment = y_values[peak_index:]
            if len(left_segment) == len(right_segment):
                feature_dict['is_abs'] = int(np.all(np.sign(left_segment) != np.sign(right_segment)) and np.all(np.abs(left_segment) == np.abs(right_segment)))
            else:
                feature_dict['is_abs'] = 0
        else:
            feature_dict['is_abs'] = 0
    else:
        feature_dict['is_abs'] = 0
    
    # Even or odd function behavior
    feature_dict['is_even'] = int(np.sign(y_values[0]) != np.sign(y_values[-1]))
    feature_dict['is_odd'] = int(np.sign(y_values[0]) == np.sign(y_values[-1]))
    
    # Sine function behavior
    if len(peaks) > 1 and len(valleys) > 1:
        max_equal = np.all(np.isclose(y_values[peaks], y_values[peaks][0]))
        min_equal = np.all(np.isclose(y_values[valleys], y_values[valleys][0]))
        feature_dict['is_sine'] = int(max_equal and min_equal)
    else:
        feature_dict['is_sine'] = 0
    
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
    
    # Verify the shape of the feature array
    print(f"Feature array shape: {features.shape}")
    
    # Debugging prints before feeding the input to the model
    logging.debug("Graph data shape: %s", features.shape)
    logging.debug("Number of features: %d", features.size)
    
    prediction = model.predict([features])
    logging.debug("Prediction: %s", prediction[0])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)