import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image
import os  # Import the os module

def yplotlimit(x, y):
    maxima = []
    minima = []
    hasmin = 0
    hasmax = 0
    combine = np.column_stack((x, y))
    for p in range(len(combine)):
        if (p == 0) or (p == 99):
            continue
        elif (combine[p][1] > combine[p-1][1]) and (combine[p][1] > combine[p+1][1]):
            maxima.append(combine[p][1])
    if maxima:
        m = max(maxima)
        hasmax = 1
    for q in range(len(combine)):
        if (q == 0) or (q == 99):
            continue
        elif (combine[q][1] < combine[q-1][1]) and (combine[q][1] < combine[q+1][1]):
            minima.append(combine[q][1])
    if minima:
        n = min(minima)
        hasmin = 1
    
    if hasmax == 1:
        mp = m + (m/20)
    if hasmin == 1:
        if n < 0:
            n *= -1
        nm = n + (n/20)
    if hasmin == 1 and hasmax == 1:
        mmmm = max(nm, mp)
        return mmmm
    elif hasmin == 1 and hasmax != 1:
        return nm
    elif hasmax == 1 and hasmin != 1:
        return mp
    else:
        return 5

def image_to_matrix(image_path, new_size=(250, 250)):
    img = imageio.imread(image_path)
    img_resized = np.array(Image.fromarray(img).resize(new_size))
    # Convert to binary matrix: path (black) as 1, background (white) as 0
    binary_matrix = (img_resized[:, :, 0] < 128).astype(int)
    return binary_matrix

def generate_random_eighthdeg_plot():
    a = np.random.uniform(-5, 5)  
    b = np.random.uniform(-5, 5)   
    c = np.random.uniform(-5, 5)  
    d = np.random.uniform(-5, 5)
    e = np.random.uniform(-5, 5)
    f = np.random.uniform(-5, 5) 
    g = np.random.uniform(-5, 5)
    h = np.random.uniform(-5, 5)
    j = np.random.uniform(-5, 5)
    while j == 0:
        j = np.random.uniform(-5, 5) 
    x_values = np.linspace(-5, 5, 100)
    y_values = []
    for z in x_values:
        y_values.append(j*(z+a)*(z+b)*(z+c)*(z+d)*(z+e)*(z+f)*(z+g)*(z+h))        
    plt.figure(figsize=(8, 8))
    plt.plot(x_values, y_values, color='k', linestyle='-')
    m = yplotlimit(x_values, y_values)
    plt.xlim(-5, 5)
    plt.ylim(-m, m)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('test_eighthdeg_function.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    # Convert image to binary matrix
    image_path = 'test_eighthdeg_function.jpg'
    matrix = image_to_matrix(image_path, new_size=(250, 250))

    return matrix

if __name__ == "__main__":
    matrix = generate_random_eighthdeg_plot()
    print("Generated graph matrix with shape:", matrix.shape)
    
    # Save the matrix as a text file
    np.savetxt('graphmatrix.txt', matrix, fmt='%d')
    print("Saved matrix as 'graphmatrix.txt'")
    
    # Load the matrix from the text file
    loaded_matrix = np.loadtxt('graphmatrix.txt', dtype=int)
    
    # Debugging statement to read out the number of features
    print("Loaded graph matrix shape:", loaded_matrix.shape)
    print("Number of features in the graph matrix:", loaded_matrix.size)