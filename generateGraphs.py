import numpy as np
import matplotlib.pyplot as plt
import os

# Define the output directory
output_dir = 'dataset'

# Define the types of functions and their corresponding labels
def linear(x, a, b): return a * x + b
def quadratic(x, a, b, c): return a * x**2 + b * x + c
def cubic(x, a, b, c, d): return a * x**3 + b * x**2 + c * x + d
def quartic(x, a, b, c, d, e): return a * x**4 + b * x**3 + c * x**2 + d * x + e
def exponential(x, a, b): return a * np.exp(b * x)
def logarithmic(x, a, b): return a * np.log(np.clip(x, 1e-10, None)) + b
def sine(x, a, b, c): return a * np.sin(b * x + c)
def cosine(x, a, b, c): return a * np.cos(b * x + c)
def tangent(x, a, b, c): return a * np.tan(b * x + c)
def absolute_value(x, a, b): return a * np.abs(x) + b
def square_root(x, a, b): return a * np.sqrt(np.clip(x, 0, None)) + b
def reciprocal(x, a, b): return a / np.clip(x, 1e-10, None) + b
def piecewise(x, a, b, c, d): return np.piecewise(x, [x < 0, x >= 0], [lambda x: a * x**2 + b, lambda x: c * np.sqrt(x) + d])

function_types = {
    'linear': linear,
    'quadratic': quadratic,
    'cubic': cubic,
    'quartic': quartic,
    'exponential': exponential,
    'logarithmic': logarithmic,
    'sine': sine,
    'cosine': cosine,
    'tangent': tangent,
    'absolute_value': absolute_value,
    'square_root': square_root,
    'reciprocal': reciprocal,
    'piecewise': piecewise
}

# Parameters for generating the graphs
num_samples = 1000
img_height, img_width = 150, 150
x = np.linspace(-10, 10, num_samples)

# Create the output directories
os.makedirs(output_dir, exist_ok=True)
for function_type in function_types.keys():
    os.makedirs(os.path.join(output_dir, 'images', function_type), exist_ok=True)

# Initialize lists to store the data and labels
data = []
labels = []

# Generate and save the graphs
def save_graphs(function_type, function, num_graphs):
    for i in range(num_graphs):
        # Add random noise to the x-values
        x_noisy = x + np.random.normal(0, 0.1, size=x.shape)

        # Generate random parameters for the function
        if function_type in ['linear', 'absolute_value', 'square_root', 'reciprocal']:
            params = np.random.uniform(-2, 2, size=2)
        elif function_type in ['quadratic']:
            params = np.random.uniform(-2, 2, size=3)
        elif function_type in ['cubic']:
            params = np.random.uniform(-2, 2, size=4)
        elif function_type in ['quartic']:
            params = np.random.uniform(-2, 2, size=5)
        elif function_type in ['exponential', 'logarithmic']:
            params = np.random.uniform(-2, 2, size=2)
        elif function_type in ['sine', 'cosine', 'tangent']:
            params = np.random.uniform(-2, 2, size=3)
        elif function_type in ['piecewise']:
            params = np.random.uniform(-2, 2, size=4)

        y = function(x_noisy, *params)
        plt.figure(figsize=(2, 2))
        plt.plot(x_noisy, y)
        plt.axis('off')
        plt.tight_layout()
        
        # Save the image
        img_path = os.path.join(output_dir, 'images', function_type, f'{function_type}_{i}.png')
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Load the image and convert to grayscale
        img = plt.imread(img_path)
        img_gray = np.mean(img, axis=2)

        # Resize the image to the desired dimensions
        img_resized = np.resize(img_gray, (img_height, img_width))

        # Append the matrix and label to the lists
        data.append(img_resized)
        labels.append(function_type)

        # Print progress
        print(f"Generated {function_type} graph {i+1}/{num_graphs}")

# Generate graphs for each function type
for function_type, function in function_types.items():
    save_graphs(function_type, function, 140)  # 100 for training, 20 for validation, 20 for testing

# Convert lists to arrays
data = np.array(data)
labels = np.array(labels)

# Save the data and labels to a single .npz file
np.savez_compressed(os.path.join(output_dir, 'graphs_dataset.npz'), data=data, labels=labels)

print("Test data generation complete.")