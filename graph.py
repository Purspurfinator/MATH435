import numpy as np
import matplotlib.pyplot as plt
import imageio as imageio
import os
import math

D = []
L = []

def yplotlimit(x,y):
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


# Function to create and save random line plots
def generate_random_line_plots(num_plots):
    for i in range(num_plots):
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

        # Add horizontal and vertical lines at zero
        plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        plt.axvline(0, color='black', linewidth=0.5, ls='solid')

        # Save the plot as a JPG image
        plt.savefig(f'random_line_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)

        # Close the plot to free memory
        plt.close()

def generate_random_quad_plots(num_plots):
    for i in range(num_plots):
        a = np.random.uniform(-5, 5)  
        b = np.random.uniform(-5, 5)   
        c = np.random.uniform(-5, 5)
        while c == 0:
            c = np.random.uniform(-5, 5)        


        # Generate x values from -5 to 5
        x_values = np.linspace(-5, 5, 100)
        # Calculate corresponding y values using the linear equation
        y_values = []
        for z in x_values:
            y_values.append(c*(z+a)*(z+b))

        # Create the plot
        plt.figure(figsize=(8, 8))
        plt.plot(x_values, y_values, color='k', linestyle='-')
        m = yplotlimit(x_values, y_values)
        # Set the limits of the plot
        plt.xlim(-5, 5)
        plt.ylim(-m, m)

        plt.xticks([])
        plt.yticks([])

        # Add horizontal and vertical lines at zero
        plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        plt.axvline(0, color='black', linewidth=0.5, ls='solid')

        # Save the plot as a JPG image
        plt.savefig(f'random_quad_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)

        # Close the plot to free memory
        plt.close()

def generate_random_cubic_plots(num_plots):
    for i in range(num_plots):
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

        # Add horizontal and vertical lines at zero
        plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        plt.axvline(0, color='black', linewidth=0.5, ls='solid')

        # Save the plot as a JPG image
        plt.savefig(f'random_cubic_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)

        # Close the plot to free memory
        plt.close()

def generate_random_fourthdeg_plots(num_plots):
    for i in range(num_plots):
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

        # Add horizontal and vertical lines at zero
        plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        plt.axvline(0, color='black', linewidth=0.5, ls='solid')

        # Save the plot as a JPG image
        plt.savefig(f'random_fourthdeg_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)

        # Close the plot to free memory
        plt.close()


def generate_random_fifthdeg_plots(num_plots):
    for i in range(num_plots):
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
        plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        plt.axvline(0, color='black', linewidth=0.5, ls='solid')
        plt.savefig(f'random_fifthdeg_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

def generate_random_sixthdeg_plots(num_plots):
    for i in range(num_plots):
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
        plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        plt.axvline(0, color='black', linewidth=0.5, ls='solid')
        plt.savefig(f'random_sixthdeg_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

def generate_random_seventhdeg_plots(num_plots):
    for i in range(num_plots):
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
        plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        plt.axvline(0, color='black', linewidth=0.5, ls='solid')
        plt.savefig(f'random_seventhdeg_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

def generate_random_eighthdeg_plots(num_plots):
    for i in range(num_plots):
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
            y_values.append(j*(z+a)*(z+b)*(z+c)*(z+d)*(z+e)*(z+f)*(z+g)*(z+h))        # Create the plot
        plt.figure(figsize=(8, 8))
        plt.plot(x_values, y_values, color='k', linestyle='-')
        m = yplotlimit(x_values, y_values)
        # Set the limits of the plot
        plt.xlim(-5, 5)
        plt.ylim(-m, m)
        plt.xticks([])
        plt.yticks([])
        plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        plt.axvline(0, color='black', linewidth=0.5, ls='solid')
        plt.savefig(f'random_eighthdeg_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
def generate_random_ninthdeg_plots(num_plots):
    for i in range(num_plots):
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
        plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        plt.axvline(0, color='black', linewidth=0.5, ls='solid')
        # Create the plot
        plt.figure(figsize=(8, 8))
        plt.plot(x_values, y_values, color='k', linestyle='-')
        m = yplotlimit(x_values, y_values)
        plt.xlim(-5, 5)
        plt.ylim(-m, m)
        plt.xticks([])
        plt.yticks([])
        #plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        #plt.axvline(0, color='black', linewidth=0.5, ls='solid')
        # Save the plot as a JPG image
        plt.savefig(f'random_ninthdeg_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
        # Close the plot to free memory
        plt.close()
        
        
        # # Set the limits of the plot
        # plt.xlim(-5, 5)
        # plt.ylim(-5, 5)
        # plt.xticks([])
        # plt.yticks([])
        # plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        # plt.axvline(0, color='black', linewidth=0.5, ls='solid')
        # plt.savefig(f'random_ninthdeg_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.close()

def generate_random_tenthdeg_plots(num_plots):
    for i in range(num_plots):
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
        #y_values = x_values**10 * j +  x_values ** 9 * i + x_values**8 * h + x_values**7 * g + x_values** 6 * f + x_values**5 * e + x_values** 3 * d + x_values**2 * a + b * x_values + c
        # Create the plot
        plt.figure(figsize=(8, 8))
        plt.plot(x_values, y_values, color='k', linestyle='-')
        m = yplotlimit(x_values, y_values)
        # Set the limits of the plot
        plt.xlim(-5, 5)
        plt.ylim(-m, m)
        plt.xticks([])
        plt.yticks([])
        plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        plt.axvline(0, color='black', linewidth=0.5, ls='solid')
        plt.savefig(f'random_tenthdeg_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

def generate_random_sin_plots(num_plots):
    for i in range(num_plots):
        a = np.random.uniform(-5, 5)  
        b = np.random.uniform(-5, 5)  
        c = np.random.uniform(-5, 5)  
        d = np.random.uniform(-5, 5) 
        while a == 0:
            a = np.random.uniform(-5,5)
        # Generate x values from -5 to 5
        x_values = np.linspace(-5, 5, 100)
        #y_values = math.sin(b*(x_values)-c) * a + d
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
        # Add horizontal and vertical lines at zero
        plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        plt.axvline(0, color='black', linewidth=0.5, ls='solid')
        # Save the plot as a JPG image
        plt.savefig(f'random_sin_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
        # Close the plot to free memory
        plt.close()


def generate_random_abs_plots(num_plots):
    for i in range(num_plots):
        a = np.random.uniform(-5, 5)  
        b = np.random.uniform(-5, 5)  
        c = np.random.uniform(-5, 5)  
        while a == 0:
            a = np.random.uniform(-5, 5)
        # Generate x values from -5 to 5
        x_values = np.linspace(-5, 5, 100)
        #currently dents, may need to increase the range (maybe 500 instead of 100?)

        #y_values = []
        #for j in x_values:
        #    y_values.append(a * np.abs(b * j )+ c) 
        y_values = a * np.abs(b * (x_values) + c)
        # Create the plot
        plt.figure(figsize=(8, 8))
        plt.plot(x_values, y_values, color='k', linestyle='-')
        # Set the limits of the plot
        #m = yplotlimit(x_values, y_values)
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.xticks([])
        plt.yticks([])
        # Add horizontal and vertical lines at zero
        plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        plt.axvline(0, color='black', linewidth=0.5, ls='solid')
        # Save the plot
        plt.savefig(f'random_abs_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
"""
def generate_random_root_plots(num_plots):
    for i in range(num_plots):
        a = np.random.uniform(-3, 3)  
        b = np.random.uniform(0, 5)  
        c = np.random.uniform(-5, 0)  
        d = np.random.uniform(-2, 2) 
        while a == 0:
            a = np.random.uniform(-5,5)
        # Generate x values from -5 to 5
        x_values = np.linspace(0, 10, 100)
        #y_values = math.sin(b*(x_values)-c) * a + d
        # Create the plot 
        y_values = []
        
        
        for z in x_values:
            y_values.append(a*(math.sqrt(b*(z-c)))+d)

        plt.figure(figsize=(8, 8))
        plt.plot(x_values, y_values, color='k', linestyle='-')
        print(a,b,c,d)
        #m = yplotlimit(x_values, y_values)
        # Set the limits of the plot
        plt.xlim(c - 10, -1*c + 10)
        print(c-1, -1*c+1)
        if d < 0:
            plt.ylim(d-10, -1*d+10)
        else:
            plt.ylim(-1*d-10, d+10)
        plt.xticks([])
        plt.yticks([])
        # Add horizontal and vertical lines at zero
        plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        plt.axvline(0, color='black', linewidth=0.5, ls='solid')
        # Save the plot as a JPG image
        plt.savefig(f'random_root_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
        # Close the plot to free memory
        plt.close()
"""
def generate_random_exp_plots(num_plots):
    for i in range(num_plots):
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
        #m = yplotlimit(x_values, y_values)
        # Set the limits of the plot
        plt.xlim(-5, 5)
        plt.ylim(-10, 10)
        plt.xticks([])
        plt.yticks([])
        # Add horizontal and vertical lines at zero
        plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        plt.axvline(0, color='black', linewidth=0.5, ls='solid')
        # Save the plot as a JPG image
        plt.savefig(f'random_exp_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
        # Close the plot to free memory
        plt.close()    

def generate_random_log_plots(num_plots):        
   for i in range(num_plots):
        a = np.random.uniform(-5, 0)
        b = np.random.uniform(-5, 5)
        c = np.random.uniform(-5, 5)
        while a == 0:
            a = np.random.uniform(-5, 5)
       # Generate x values from -5 to 5
        x_values = np.linspace(-5, 5, 100)
        y_values = a * np.exp(b * x_values) + c
       # Calculate y values using the log function

       # Create the plot
        plt.figure(figsize=(8, 8))
        plt.plot(x_values, y_values, color='k', linestyle='-')
        #m = yplotlimit(x_values, y_values)
       # Set the limits of the plot
        plt.xlim(-5, 5)
        plt.ylim(-10, 10)
        plt.xticks([])
        plt.yticks([])
        print(a, b, c)
       # Add horizontal and vertical lines at zero
        plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        plt.axvline(0, color='black', linewidth=0.5, ls='solid')
       # Save the plot as a JPG image
        plt.savefig(f'random_log_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
       # Close the plot to free memory
        plt.close()      

def image_to_matrix(image_path, new_size=(200, 200)):
    img = imageio.imread(image_path)
    
    original_height, original_width = img.shape[:2]

    img_resized = np.zeros((new_size[0], new_size[1], img.shape[2]), dtype=img.dtype)

    height_scale = original_height / new_size[0]
    width_scale = original_width / new_size[1]

    for i in range(new_size[0]):
        for j in range(new_size[1]):
            orig_x = int(j * width_scale)
            orig_y = int(i * height_scale)
            img_resized[i, j] = img[orig_y, orig_x]

    return img_resized


#num_plots = int(input("Enter the number of random line plots to generate: "))
number = int(input("Enter amount of graphs to generate: "))
for i in range(number):
    generate_random_line_plots(1) 
    image_path = 'random_line_function_1.jpg'  
    matrix = image_to_matrix(image_path, new_size=(200, 200)) 
    print("Resized Matrix shape:", matrix.shape)
    D.append(matrix)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig(f'image.jpg', format='jpg')
    os.remove("random_line_function_1.jpg")
    os.remove("image.jpg")
    plt.close()
    L.append("line")
    generate_random_quad_plots(1)  
    image_path = 'random_quad_function_1.jpg'  
    matrix = image_to_matrix(image_path, new_size=(200, 200)) 
    print("Resized Matrix shape:", matrix.shape)
    D.append(matrix)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig(f'image.jpg', format='jpg')
    os.remove("random_quad_function_1.jpg")
    os.remove("image.jpg")
    plt.close()
    L.append("quad")
    generate_random_cubic_plots(1)  
    image_path = 'random_cubic_function_1.jpg'  
    matrix = image_to_matrix(image_path, new_size=(200, 200)) 
    print("Resized Matrix shape:", matrix.shape)
    D.append(matrix)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig(f'image.jpg', format='jpg')
    os.remove("random_cubic_function_1.jpg")
    os.remove("image.jpg")
    plt.close()
    L.append("cubic")
    generate_random_fourthdeg_plots(1)  
    image_path = 'random_fourthdeg_function_1.jpg'  
    matrix = image_to_matrix(image_path, new_size=(200, 200)) 
    print("Resized Matrix shape:", matrix.shape)
    D.append(matrix)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig(f'image.jpg', format='jpg')
    os.remove("random_fourthdeg_function_1.jpg")
    os.remove("image.jpg")
    plt.close()
    L.append("fourthdeg")
    generate_random_fifthdeg_plots(1)  
    image_path = 'random_fifthdeg_function_1.jpg' 
    matrix = image_to_matrix(image_path, new_size=(200, 200)) 
    print("Resized Matrix shape:", matrix.shape)
    D.append(matrix)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig(f'image.jpg', format='jpg')
    os.remove("random_fifthdeg_function_1.jpg")
    os.remove("image.jpg")
    plt.close()
    L.append("fifthdeg")
    generate_random_sixthdeg_plots(1)  
    image_path = 'random_sixthdeg_function_1.jpg' 
    matrix = image_to_matrix(image_path, new_size=(200, 200)) 
    print("Resized Matrix shape:", matrix.shape)
    D.append(matrix)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig(f'image.jpg', format='jpg')
    os.remove("random_sixthdeg_function_1.jpg")
    os.remove("image.jpg")
    plt.close()
    L.append("sixthdeg")
    generate_random_seventhdeg_plots(1)  
    image_path = 'random_seventhdeg_function_1.jpg'  
    matrix = image_to_matrix(image_path, new_size=(200, 200)) 
    print("Resized Matrix shape:", matrix.shape)
    D.append(matrix)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig(f'image.jpg', format='jpg')
    os.remove("random_seventhdeg_function_1.jpg")
    os.remove("image.jpg")
    plt.close()
    L.append("seventhdeg")
    generate_random_eighthdeg_plots(1)  
    image_path = 'random_eighthdeg_function_1.jpg'  
    matrix = image_to_matrix(image_path, new_size=(200, 200)) 
    print("Resized Matrix shape:", matrix.shape)
    D.append(matrix)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig(f'image.jpg', format='jpg')
    os.remove("random_eighthdeg_function_1.jpg")
    os.remove("image.jpg")
    plt.close()
    L.append("eighthdeg")
    generate_random_ninthdeg_plots(1)  
    image_path = 'random_ninthdeg_function_1.jpg'  
    matrix = image_to_matrix(image_path, new_size=(200, 200)) 
    print("Resized Matrix shape:", matrix.shape)
    D.append(matrix)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig(f'image.jpg', format='jpg')
    os.remove("random_ninthdeg_function_1.jpg")
    os.remove("image.jpg")
    plt.close()
    L.append("ninthdeg")
    generate_random_tenthdeg_plots(1)  
    image_path = 'random_tenthdeg_function_1.jpg'  
    matrix = image_to_matrix(image_path, new_size=(200, 200)) 
    print("Resized Matrix shape:", matrix.shape)
    D.append(matrix)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig(f'image.jpg', format='jpg')
    os.remove("random_tenthdeg_function_1.jpg")
    os.remove("image.jpg")
    plt.close()
    L.append("tenthdeg")
    generate_random_sin_plots(1)  
    image_path = 'random_sin_function_1.jpg'  
    matrix = image_to_matrix(image_path, new_size=(200, 200)) 
    print("Resized Matrix shape:", matrix.shape)
    D.append(matrix)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig(f'image.jpg', format='jpg')
    os.remove("random_sin_function_1.jpg")
    os.remove("image.jpg")
    plt.close()
    L.append("sin")
    generate_random_abs_plots(1)  
    image_path = 'random_abs_function_1.jpg'  
    matrix = image_to_matrix(image_path, new_size=(200, 200)) 
    print("Resized Matrix shape:", matrix.shape)
    D.append(matrix)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig(f'image.jpg', format='jpg')
    os.remove("random_abs_function_1.jpg")
    os.remove("image.jpg")
    plt.close()
    L.append("abs")
    generate_random_exp_plots(1)  
    image_path = 'random_exp_function_1.jpg'  
    matrix = image_to_matrix(image_path, new_size=(200, 200)) 
    print("Resized Matrix shape:", matrix.shape)
    D.append(matrix)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig(f'image.jpg', format='jpg')
    os.remove("random_exp_function_1.jpg")
    os.remove("image.jpg")
    plt.close()
    L.append("exp")
    """
    generate_random_log_plots(1)  
    image_path = 'random_log_function_1.jpg'  
    matrix = image_to_matrix(image_path, new_size=(200, 200)) 
    print("Resized Matrix shape:", matrix.shape)
    D.append(matrix)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig(f'image.jpg', format='jpg')
    #os.remove("random_log_function_1.jpg")
    os.remove("image.jpg")
    L.append("log")
    
    generate_random_root_plots(1)  
    image_path = 'random_root_function_1.jpg'  
    matrix = image_to_matrix(image_path, new_size=(200, 200)) 
    print("Resized Matrix shape:", matrix.shape)
    D.append(matrix)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig(f'image.jpg', format='jpg')
    #os.remove("random_root_function_1.jpg")
    os.remove("image.jpg")
    L.append("root") 
    """



np.save('Matricies.npy', D)
np.save('Labels.npy', L)
