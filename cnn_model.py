from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

def cnn_model():

#loading dataset
    data = np.load('Matrices.npy')
    labels = np.load('Labels.npy')
    #check if data = grayscale 150x150 and labels = 1 dimensional with class labels
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    # Flatten the 2D matrices to 1D arrays for the model
    data = data.reshape(data.shape[0], 150, 150, 1)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

#exploring data

    plt.imshow(X_train[0].reshape(150, 150), cmap="gray")
    plt.show()

#building the model

    model = sequential.Sequential()
    model.add(Conv2D(32,kernel_size=3, activation='relu',input_shape=(150,150,1)))
    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(Conv2D(128,kernel_size=3,activation='relu'))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(14,activation='softmax'))

    return model, X_train, X_test, y_train, y_test

#compiling the model

model, X_train, X_test, y_train, y_test = cnn_model()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

X_train = X_train/255.0
X_test = X_test/255.0

y_train = to_categorical(y_train, num_classes=14)
y_test = to_categorical(y_test, num_classes=14)

#training the model

history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10, batch_size = 32)

#evaluate model

test_loss, test_accuracy = model.evaluate(X_test,y_test)
print(test_accuracy)

# plot training history

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Training and Validation Accuracy")
plt.show()