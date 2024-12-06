import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm

# Step 1: Load and preprocess the data
def load_data():
    data = np.load('Matricies.npy')
    labels = np.load('Labels.npy')
    # Flatten the 2D matrices to 1D arrays for the model
    data = data.reshape(data.shape[0], -1)
    return data, labels

# Step 2: Train the model
def train_model(data, labels):
    # Shuffle the data and labels together
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

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
    train_model(data, labels)
