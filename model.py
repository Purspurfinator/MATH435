import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm

# Step 1: Load and preprocess the data
def load_data():
    data = np.load('Matrices.npy')
    labels = np.load('Labels.npy')
    
    # Ensure the data is reshaped to 200x200 pixels
    data = data.reshape(data.shape[0], 200, 200)
    
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

        features.append([num_maxima, num_minima])
    
    return np.array(features)

# Step 3: Train the model
def train_model(features, labels):
    # Shuffle the data and labels together
    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)
    features = features[indices]
    labels = labels[indices]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train a single model with progress output
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Add progress bar for the training process
    for i in tqdm(range(1, 101), desc="Training Progress"):
        model.n_estimators = i
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")
    joblib.dump(model, 'graph_model.pkl')

if __name__ == "__main__":
    data, labels = load_data()
    features = extract_features(data)
    train_model(features, labels)
