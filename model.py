import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

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
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")
    joblib.dump(model, 'graph_model.pkl')

# Step 3: Load the trained model and make predictions
def predict_graph(graph_data):
    model = joblib.load('graph_model.pkl')
    graph_data = graph_data.reshape(1, -1)  # Flatten the 2D matrix to 1D array
    prediction = model.predict(graph_data)
    return prediction

if __name__ == "__main__":
    data, labels = load_data()
    train_model(data, labels)
    
    # Load test data from test.npy
    test_data = np.load('test.npy')
    # Flatten the 2D matrix to 1D array
    test_data = test_data.reshape(1, -1)
    
    # Make a prediction on the test data
    prediction = predict_graph(test_data)
    print(f"Predicted graph type: {prediction}")
