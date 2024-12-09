import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # Updated import statement
import os
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Lock
from PIL import Image

D = []
L = []

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
        else:
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

def generate_random_line_plot(i, progress, lock, graph_type):
    m = np.random.uniform(-7, 7)  # Random slope
    b = np.random.uniform(-5, 5)   # Random intercept

    # Generate x values from -5 to 5
    x_values = np.linspace(-5, 5, 100)
    # Calculate corresponding y values using the linear equation
    y_values = m * x_values + b

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.plot(x_values, y_values, color='k', linestyle='-')
    m = yplotlimit(x_values, y_values)
    # Set the limits of the plot
    plt.xlim(-5, 5)
    plt.ylim(-m, m)

    plt.xticks([])
    plt.yticks([])

    # Save the plot as a JPG image
    plt.savefig(f'random_line_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)

    # Close the plot to free memory
    plt.close()

    # Convert image to matrix
    image_path = f'random_line_function_{i + 1}.jpg'
    matrix = image_to_matrix(image_path, new_size=(200, 200))
    os.remove(image_path)

    with lock:
        progress.value += 1
        print(f"Generated {progress.value} graphs ({graph_type})")

    return matrix, "line"

def generate_random_quad_plot(i, progress, lock, graph_type):
    a = np.random.uniform(-5, 5)  
    b = np.random.uniform(-5, 5)   
    c = np.random.uniform(-5, 5)
    while c == 0:
        c = np.random.uniform(-5, 5)        

    # Generate x values from -5 to 5
    x_values = np.linspace(-5, 5, 100)
    # Calculate corresponding y values using the quadratic equation
    y_values = c * (x_values + a) * (x_values + b)

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.plot(x_values, y_values, color='k', linestyle='-')
    m = yplotlimit(x_values, y_values)
    # Set the limits of the plot
    plt.xlim(-5, 5)
    plt.ylim(-m, m)

    plt.xticks([])
    plt.yticks([])

    # Save the plot as a JPG image
    plt.savefig(f'random_quad_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)

    # Close the plot to free memory
    plt.close()

    # Convert image to matrix
    image_path = f'random_quad_function_{i + 1}.jpg'
    matrix = image_to_matrix(image_path, new_size=(200, 200))
    os.remove(image_path)

    with lock:
        progress.value += 1
        print(f"Generated {progress.value} graphs ({graph_type})")

    return matrix, "quad"

def generate_random_cubic_plot(i, progress, lock, graph_type):
    a = np.random.uniform(-5, 5)  
    b = np.random.uniform(-5, 5)   
    c = np.random.uniform(-5, 5)  
    d = np.random.uniform(-5, 5)
    while d == 0:
        d = np.random.uniform(-5, 5)  


    # Generate x values from -5 to 5
    x_values = np.linspace(-5, 5, 100)
    # Calculate corresponding y values using the linear equation
    #y_values = x_values** 3 * d + x_values**2 * a + b * x_values + c
    y_values = []
    for z in x_values:
        y_values.append(d*(z+a)*(z+b)*(z+c))

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.plot(x_values, y_values, color='k', linestyle='-')
    m = yplotlimit(x_values, y_values)
    # Set the limits of the plot
    plt.xlim(-5, 5)
    plt.ylim(-m, m)

    plt.xticks([])
    plt.yticks([])


    # Save the plot as a JPG image
    plt.savefig(f'random_cubic_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)

    # Close the plot to free memory
    plt.close()

    # Convert image to matrix
    image_path = f'random_cubic_function_{i + 1}.jpg'
    matrix = image_to_matrix(image_path, new_size=(200, 200))
    os.remove(image_path)

    with lock:
        progress.value += 1
        print(f"Generated {progress.value} graphs ({graph_type})")

    return matrix, "cubic"

def generate_random_fourthdeg_plot(i, progress, lock, graph_type):
    a = np.random.uniform(-5, 5)  
    b = np.random.uniform(-5, 5)   
    c = np.random.uniform(-5, 5)  
    d = np.random.uniform(-5, 5) 
    e = np.random.uniform(-5, 5)
    while e == 0:
        e = np.random.uniform(-5, 5)  


    # Generate x values from -5 to 5
    x_values = np.linspace(-5, 5, 100)
    # Calculate corresponding y values using the linear equation
    #y_values = x_values**4 * e + x_values** 3 * d + x_values**2 * a + b * x_values + c
    y_values = []
    for z in x_values:
        y_values.append(e*(z+a)*(z+b)*(z+c)*(z+d))

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.plot(x_values, y_values, color='k', linestyle='-')
    m = yplotlimit(x_values, y_values)
    # Set the limits of the plot
    plt.xlim(-5, 5)
    plt.ylim(-m, m)

    plt.xticks([])
    plt.yticks([])
    

    # Save the plot as a JPG image
    plt.savefig(f'random_fourthdeg_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)

    # Close the plot to free memory
    plt.close()

    # Convert image to matrix
    image_path = f'random_fourthdeg_function_{i + 1}.jpg'
    matrix = image_to_matrix(image_path, new_size=(200, 200))
    os.remove(image_path)

    with lock:
        progress.value += 1
        print(f"Generated {progress.value} graphs ({graph_type})")

    return matrix, "fourthdeg"

def generate_random_fifthdeg_plot(i, progress, lock, graph_type):
    a = np.random.uniform(-5, 5)  
    b = np.random.uniform(-5, 5)   
    c = np.random.uniform(-5, 5)  
    d = np.random.uniform(-5, 5)
    e = np.random.uniform(-5, 5) 
    f = np.random.uniform(-5, 5)
    while f == 0:
        f = np.random.uniform(-5, 5)
    x_values = np.linspace(-5, 5, 100)
    y_values = []
    for z in x_values:
        y_values.append(f*(z+a)*(z+b)*(z+c)*(z+d)*(z+e))
    plt.figure(figsize=(8, 8))
    plt.plot(x_values, y_values, color='k', linestyle='-')
    m = yplotlimit(x_values, y_values)
    plt.xlim(-5, 5)
    plt.ylim(-m, m)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'random_fifthdeg_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    # Convert image to matrix
    image_path = f'random_fifthdeg_function_{i + 1}.jpg'
    matrix = image_to_matrix(image_path, new_size=(200, 200))
    os.remove(image_path)

    with lock:
        progress.value += 1
        print(f"Generated {progress.value} graphs ({graph_type})")

    return matrix, "fifthdeg"

def generate_random_sixthdeg_plot(i, progress, lock, graph_type):
    a = np.random.uniform(-5, 5)  
    b = np.random.uniform(-5, 5)   
    c = np.random.uniform(-5, 5)  
    d = np.random.uniform(-5, 5)
    e = np.random.uniform(-5, 5)
    f = np.random.uniform(-5, 5)
    g = np.random.uniform(-5, 5)
    while a == 0:
        a = np.random.uniform(-5, 5) 
    x_values = np.linspace(-5, 5, 100)
    y_values = []
    for z in x_values:
        y_values.append(g*(z+a)*(z+b)*(z+c)*(z+d)*(z+e)*(z+f))
    plt.figure(figsize=(8, 8))
    plt.plot(x_values, y_values, color='k', linestyle='-')
    m = yplotlimit(x_values, y_values)
    plt.xlim(-5, 5)
    plt.ylim(-m, m)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'random_sixthdeg_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    # Convert image to matrix
    image_path = f'random_sixthdeg_function_{i + 1}.jpg'
    matrix = image_to_matrix(image_path, new_size=(200, 200))
    os.remove(image_path)

    with lock:
        progress.value += 1
        print(f"Generated {progress.value} graphs ({graph_type})")

    return matrix, "sixthdeg"

def generate_random_seventhdeg_plot(i, progress, lock, graph_type):
    a = np.random.uniform(-5, 5)  
    b = np.random.uniform(-5, 5)   
    c = np.random.uniform(-5, 5)  
    d = np.random.uniform(-5, 5)
    e = np.random.uniform(-5, 5)
    f = np.random.uniform(-5, 5) 
    g = np.random.uniform(-5, 5)
    h = np.random.uniform(-5, 5)
    while g == 0:
        g = np.random.uniform(-5, 5) 
    x_values = np.linspace(-5, 5, 100)
    y_values = []
    for z in x_values:
        y_values.append(g*(z+a)*(z+b)*(z+c)*(z+d)*(z+e)*(z+f)*(z+h))
    plt.figure(figsize=(8, 8))
    plt.plot(x_values, y_values, color='k', linestyle='-')
    m = yplotlimit(x_values, y_values)
    plt.xlim(-5, 5)
    plt.ylim(-m, m)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'random_seventhdeg_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    # Convert image to matrix
    image_path = f'random_seventhdeg_function_{i + 1}.jpg'
    matrix = image_to_matrix(image_path, new_size=(200, 200))
    os.remove(image_path)

    with lock:
        progress.value += 1
        print(f"Generated {progress.value} graphs ({graph_type})")

    return matrix, "seventhdeg"

def generate_random_eighthdeg_plot(i, progress, lock, graph_type):
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
    # Generate x values from -5 to 5
    x_values = np.linspace(-5, 5, 100)
    # Calculate corresponding y values using the linear equation
    y_values = []
    for z in x_values:
        y_values.append(j*(z+a)*(z+b)*(z+c)*(z+d)*(z+e)*(z+f)*(z+g)*(z+h))        
    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.plot(x_values, y_values, color='k', linestyle='-')
    m = yplotlimit(x_values, y_values)
    # Set the limits of the plot
    plt.xlim(-5, 5)
    plt.ylim(-m, m)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'random_eighthdeg_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    # Convert image to matrix
    image_path = f'random_eighthdeg_function_{i + 1}.jpg'
    matrix = image_to_matrix(image_path, new_size=(200, 200))
    os.remove(image_path)

    with lock:
        progress.value += 1
        print(f"Generated {progress.value} graphs ({graph_type})")

    return matrix, "eighthdeg"

def generate_random_ninthdeg_plot(i, progress, lock, graph_type):
    a = np.random.uniform(-5, 5)  
    b = np.random.uniform(-5, 5)   
    c = np.random.uniform(-5, 5)  
    d = np.random.uniform(-5, 5)
    e = np.random.uniform(-5, 5)
    f = np.random.uniform(-5, 5) 
    g = np.random.uniform(-5, 5)
    h = np.random.uniform(-5, 5)
    j = np.random.uniform(-5, 5)
    k = np.random.uniform(-5, 5)
    while j == 0:
        j = np.random.uniform(-5, 5) 
    
    # Generate x values from -5 to 5
    x_values = np.linspace(-5, 5, 100)
    # Calculate corresponding y values using the linear equation
    y_values = []
    for z in x_values:
        y_values.append(j*(z+a)*(z+b)*(z+c)*(z+d)*(z+e)*(z+f)*(z+g)*(z+h)*(z+k))
    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.plot(x_values, y_values, color='k', linestyle='-')
    m = yplotlimit(x_values, y_values)
    plt.xlim(-5, 5)
    plt.ylim(-m, m)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'random_ninthdeg_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    # Convert image to matrix
    image_path = f'random_ninthdeg_function_{i + 1}.jpg'
    matrix = image_to_matrix(image_path, new_size=(200, 200))
    os.remove(image_path)

    with lock:
        progress.value += 1
        print(f"Generated {progress.value} graphs ({graph_type})")

    return matrix, "ninthdeg"

def generate_random_tenthdeg_plot(i, progress, lock, graph_type):
    a = np.random.uniform(-5, 5)  
    b = np.random.uniform(-5, 5)   
    c = np.random.uniform(-5, 5)  
    d = np.random.uniform(-5, 5)
    e = np.random.uniform(-5, 5)
    f = np.random.uniform(-5, 5) 
    g = np.random.uniform(-5, 5)
    h = np.random.uniform(-5, 5)
    k = np.random.uniform(-5, 5)
    j = np.random.uniform(-5, 5)
    l = np.random.uniform(-5,5)
    while j == 0:
        j = np.random.uniform(-5, 5) 
    # Generate x values from -5 to 5
    x_values = np.linspace(-5, 5, 100)
    # Calculate corresponding y values using the linear equation
    y_values = []
    for z in x_values:
        y_values.append(j*(z+a)*(z+b)*(z+c)*(z+d)*(z+e)*(z+f)*(z+g)*(z+h)*(z+k)*(z+l))
    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.plot(x_values, y_values, color='k', linestyle='-')
    m = yplotlimit(x_values, y_values)
    # Set the limits of the plot
    plt.xlim(-5, 5)
    plt.ylim(-m, m)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'random_tenthdeg_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    # Convert image to matrix
    image_path = f'random_tenthdeg_function_{i + 1}.jpg'
    matrix = image_to_matrix(image_path, new_size=(200, 200))
    os.remove(image_path)

    with lock:
        progress.value += 1
        print(f"Generated {progress.value} graphs ({graph_type})")

    return matrix, "tenthdeg"

def generate_random_sin_plot(i, progress, lock, graph_type):
    a = np.random.uniform(-5, 5)  
    b = np.random.uniform(-5, 5)  
    c = np.random.uniform(-5, 5)  
    d = np.random.uniform(-5, 5) 
    while a == 0:
        a = np.random.uniform(-5,5)
    # Generate x values from -5 to 5
    x_values = np.linspace(-5, 5, 100)
    # Create the plot 
    y_values = []
    for j in x_values:
        y_values.append(math.sin(b*(j)-c) * a + d)
    plt.figure(figsize=(8, 8))
    plt.plot(x_values, y_values, color='k', linestyle='-')
    m = yplotlimit(x_values, y_values)
    # Set the limits of the plot
    plt.xlim(-5, 5)
    plt.ylim(-m, m)
    plt.xticks([])
    plt.yticks([])

    # Save the plot as a JPG image
    plt.savefig(f'random_sin_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    # Close the plot to free memory
    plt.close()

    # Convert image to matrix
    image_path = f'random_sin_function_{i + 1}.jpg'
    matrix = image_to_matrix(image_path, new_size=(200, 200))
    os.remove(image_path)

    with lock:
        progress.value += 1
        print(f"Generated {progress.value} graphs ({graph_type})")

    return matrix, "sin"

def generate_random_abs_plot(i, progress, lock, graph_type):
    a = np.random.uniform(-5, 5)  
    b = np.random.uniform(-5, 5)  
    c = np.random.uniform(-5, 5)  
    while a == 0:
        a = np.random.uniform(-5, 5)
    # Generate x values from -5 to 5
    x_values = np.linspace(-5, 5, 100)
    y_values = a * np.abs(b * (x_values) + c)
    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.plot(x_values, y_values, color='k', linestyle='-')
    # Set the limits of the plot
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xticks([])
    plt.yticks([])
    # Save the plot
    plt.savefig(f'random_abs_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    # Convert image to matrix
    image_path = f'random_abs_function_{i + 1}.jpg'
    matrix = image_to_matrix(image_path, new_size=(200, 200))
    os.remove(image_path)

    with lock:
        progress.value += 1
        print(f"Generated {progress.value} graphs ({graph_type})")

    return matrix, "abs"

def generate_random_exp_plot(i, progress, lock, graph_type):
    a = np.random.uniform(0, 5)
    b = np.random.uniform(-5, 5)
    c = np.random.uniform(-5, 5)
    while a == 0:
        a = np.random.uniform(-5, 5)
    # Generate x values from -5 to 5
    x_values = np.linspace(-5, 5, 100)
    # Calculate y values using the exponential function
    y_values = a * np.exp(b * x_values) + c
    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.plot(x_values, y_values, color='k', linestyle='-')
    # Set the limits of the plot
    plt.xlim(-5, 5)
    plt.ylim(-10, 10)
    plt.xticks([])
    plt.yticks([])

    # Save the plot as a JPG image
    plt.savefig(f'random_exp_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    # Close the plot to free memory
    plt.close()    

    # Convert image to matrix
    image_path = f'random_exp_function_{i + 1}.jpg'
    matrix = image_to_matrix(image_path, new_size=(200, 200))
    os.remove(image_path)

    with lock:
        progress.value += 1
        print(f"Generated {progress.value} graphs ({graph_type})")

    return matrix, "exp"

def image_to_matrix(image_path, new_size=(200, 200)):
    img = imageio.imread(image_path)
    img_resized = np.array(Image.fromarray(img).resize(new_size))
    # Convert to binary matrix: path (black) as 1, background (white) as 0
    binary_matrix = (img_resized[:, :, 0] < 128).astype(int)
    return binary_matrix

def generate_graphs(graph_type, num_graphs, progress, lock, max_workers):
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(graph_type, i, progress, lock, graph_type.__name__) for i in range(num_graphs)]
        for future in as_completed(futures):
            results.append(future.result())
    return results

if __name__ == "__main__":
    number = int(input("Enter amount of graphs to generate: "))
    start_time = time.time()

    manager = Manager()
    progress = manager.Value('i', 0)
    lock = manager.Lock()

    graph_types = [
        generate_random_line_plot,
        generate_random_quad_plot,
        generate_random_cubic_plot,
        generate_random_fourthdeg_plot,
        generate_random_fifthdeg_plot,
        generate_random_sixthdeg_plot,
        generate_random_seventhdeg_plot,
        generate_random_eighthdeg_plot,
        generate_random_ninthdeg_plot,
        generate_random_tenthdeg_plot,
        generate_random_sin_plot,
        generate_random_abs_plot,
        generate_random_exp_plot
    ]

    num_graphs_per_type = number // len(graph_types)
    remainder = number % len(graph_types)

    max_workers = os.cpu_count()  # Dynamically set the number of workers based on available CPU cores

    for graph_type in graph_types:
        results = generate_graphs(graph_type, num_graphs_per_type, progress, lock, max_workers)
        for matrix, label in results:
            D.append(matrix)
            L.append(label)

    # Handle the remainder
    for i in range(remainder):
        graph_type = graph_types[i % len(graph_types)]
        results = generate_graphs(graph_type, 1, progress, lock, max_workers)
        for matrix, label in results:
            D.append(matrix)
            L.append(label)

    # Save results
    np.save('Matrices.npy', D)
    np.save('Labels.npy', L)

    elapsed_time = time.time() - start_time
    print(f"Generated {number} graphs in {elapsed_time:.2f} seconds")
