import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm
from scipy.fftpack import fft

# Step 1: Load and preprocess the data
def load_data():
    data = np.load('Matrices.npy')
    labels = np.load('Labels.npy')
    
    # Debugging statements to check the shape of the loaded data
    print(f"Original data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Flatten the 2D matrices to 1D arrays
    data = data.reshape(data.shape[0], -1)
    
    return data, labels

# Step 2: Extract features
def extract_features(data):
    features = []
    for graph in data:
        y_values = graph.mean(axis=0)  # Average the pixel values along the y-axis

        maxima = []
        minima = []
        for i in range(1, len(y_values) - 1):
            if y_values[i] > y_values[i - 1] and y_values[i] > y_values[i + 1]:
                maxima.append(y_values[i])
            if y_values[i] < y_values[i - 1] and y_values[i] < y_values[i + 1]:
                minima.append(y_values[i])

        num_maxima = len(maxima)
        num_minima = len(minima)
        std_dev = np.std(y_values)
        fft_coeffs = np.abs(fft(y_values))[:10]  # Take first 10 Fourier coefficients

        features.append([num_maxima, num_minima, std_dev] + list(fft_coeffs))
    
    return np.array(features)

# Step 3: Train the model
def train_model(features, labels):
    # Shuffle the data and labels together
    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)
    features = features[indices]
    labels = labels[indices]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Define the model and hyperparameters
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform Grid Search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")
    joblib.dump(best_model, 'graph_model.pkl')

if __name__ == "__main__":
    data, labels = load_data()
    features = extract_features(data)
    train_model(features, labels)
