from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS
import logging
import base64
from PIL import Image
import io
import imageio.v2 as imageio

app = Flask(__name__)

# Update CORS configuration to allow requests from your GitHub Pages domain
CORS(app, resources={r"/*": {"origins": "https://purspurfinator.github.io"}})

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model
model = joblib.load('graph_model.pkl')

def image_to_matrix(image_path, new_size=(250, 250)):
    img = imageio.imread(image_path)
    img_rgba = Image.fromarray(img).convert('RGBA').resize(new_size)
    
    # Create a white background image
    white_bg = Image.new('RGBA', img_rgba.size, (255, 255, 255, 255))
    
    # Composite the image with the white background
    img_composite = Image.alpha_composite(white_bg, img_rgba).convert('L')
    
    # Convert to binary matrix: path (black) as 1, background (white) as 0
    binary_matrix = (np.array(img_composite) < 128).astype(int)
    
    # Debugging: Print the shape and type of the resized image
    print(f"Resized image shape: {img_composite.size}")
    print(f"Resized image type: {np.array(img_composite).dtype}")
    
    # Debugging: Print a sample of the binary matrix
    print(f"Binary matrix sample:\n{binary_matrix[:5, :5]}")
    
    return binary_matrix

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
    
    # Flatten the array for prediction
    graph_data_flattened = graph_matrix.reshape(1, -1)
    
    # Verify the shape of the flattened array
    print(f"Flattened array shape: {graph_data_flattened.shape}")
    
    # Debugging prints before feeding the input to the model
    logging.debug("Graph data shape: %s", graph_data_flattened.shape)
    logging.debug("Number of features: %d", graph_data_flattened.size)
    
    prediction = model.predict(graph_data_flattened)
    logging.debug("Prediction: %s", prediction[0])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)