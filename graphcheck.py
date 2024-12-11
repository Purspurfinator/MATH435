import numpy as np

# Load the data
data = np.load('Matrices.npy')
labels = np.load('Labels.npy')

# Inspect the shapes
data_shape = data.shape
labels_shape = labels.shape

# Inspect the data types
data_dtype = data.dtype
labels_dtype = labels.dtype

# Calculate the sizes in bytes
data_size_bytes = data.nbytes
labels_size_bytes = labels.nbytes

# Print the information
print(f"Data shape: {data_shape}")
print(f"Data dtype: {data_dtype}")
print(f"Data size (bytes): {data_size_bytes / (1024 ** 2):.2f} MB")  # Convert to MB

print(f"Labels shape: {labels_shape}")
print(f"Labels dtype: {labels_dtype}")
print(f"Labels size (bytes): {labels_size_bytes / (1024 ** 2):.2f} MB")  # Convert to MB