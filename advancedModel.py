import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from scipy.signal import find_peaks
from tqdm import tqdm
import time

def extract_features(x_values, y_values):
    features = {}
    
    # Calculate slope
    slopes = np.gradient(y_values, x_values)
    features['mean_slope'] = np.mean(slopes)
    features['std_slope'] = np.std(slopes)
    
    # Slopes at the beginning and end of the plot
    features['start_slope'] = slopes[0]
    features['end_slope'] = slopes[-1]
    
    # Find peaks (local maxima)
    peaks, _ = find_peaks(y_values)
    features['num_peaks'] = len(peaks)
    features['mean_peak_height'] = np.mean(y_values[peaks]) if len(peaks) > 0 else 0
    
    # Find valleys (local minima)
    valleys, _ = find_peaks(-y_values)
    features['num_valleys'] = len(valleys)
    features['mean_valley_depth'] = np.mean(y_values[valleys]) if len(valleys) > 0 else 0
    
    # Calculate curvature
    curvature = np.gradient(slopes, x_values)
    features['mean_curvature'] = np.mean(curvature)
    features['std_curvature'] = np.std(curvature)
    
    # Symmetry
    features['symmetry'] = np.sum(np.abs(y_values - y_values[::-1])) / len(y_values)
    
    # Periodicity (for sine functions)
    autocorr = np.correlate(y_values, y_values, mode='full')
    features['periodicity'] = np.max(autocorr[len(autocorr)//2:])
    
    # Inflection points
    inflection_points = np.where(np.diff(np.sign(curvature)))[0]
    features['num_inflection_points'] = len(inflection_points)
    
    # Amplitude and frequency (for sine functions)
    if len(peaks) > 1:
        features['amplitude'] = (np.max(y_values[peaks]) - np.min(y_values[valleys])) / 2
        features['frequency'] = len(peaks) / (x_values[-1] - x_values[0])
    else:
        features['amplitude'] = 0
        features['frequency'] = 0
    
    # Exponential growth/decay
    if np.all(y_values > 0):
        features['exp_growth_rate'] = np.mean(np.diff(np.log(y_values)))
    else:
        features['exp_growth_rate'] = 0
    
    # Specific logic for different types of plots
    features['is_parabola'] = int(np.all(np.diff(np.sign(np.gradient(slopes))) != 0))  # Quadratic plot
    features['end_behavior'] = np.sign(y_values[0]) * np.sign(y_values[-1])  # Odd degree polynomial
    features['even_end_behavior'] = int(np.sign(y_values[0]) == np.sign(y_values[-1]))  # Even degree polynomial
    
    # Maximum number of critical points for polynomials
    features['num_critical_points'] = len(peaks) + len(valleys)
    
    # Absolute value function behavior
    if len(peaks) == 1 and len(valleys) == 0:
        features['abs_slope_diff'] = abs(slopes[0] - slopes[-1])
    else:
        features['abs_slope_diff'] = 0
    
    return features

def augment_data(x_values, y_values):
    # Add random noise
    noise = np.random.normal(0, 0.1, y_values.shape)
    y_values_noisy = y_values + noise
    
    # Slightly distort the x values
    x_values_distorted = x_values + np.random.normal(0, 0.1, x_values.shape)
    
    return x_values_distorted, y_values_noisy

def load_data():
    # Load the data generated by graphs.py
    data = np.load('Matrices.npy')
    labels = np.load('Labels.npy')
    
    # Extract features for each plot
    feature_list = []
    start_time = time.time()
    for i in range(data.shape[0]):
        x_values = np.linspace(-5, 5, 100)
        y_values = data[i].reshape(250, 250).mean(axis=1)  # Simplified example
        
        # Augment data
        x_values_aug, y_values_aug = augment_data(x_values, y_values)
        
        features = extract_features(x_values_aug, y_values_aug)
        feature_list.append(features)
    end_time = time.time()
    print(f"Feature extraction time: {end_time - start_time} seconds")
    
    # Convert feature list to a numpy array
    feature_array = np.array([list(f.values()) for f in feature_list])
    
    return feature_array, labels

def train_advanced_model(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Add progress bar for the training process
    start_time = time.time()
    for i in tqdm(range(1, 101), desc="Training Progress"):
        model.n_estimators = i
        model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Advanced Model accuracy: {accuracy}")
    
    joblib.dump(model, 'advanced_graph_model.pkl')

if __name__ == "__main__":
    data, labels = load_data()
    train_advanced_model(data, labels)
