import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

D = []
L = []



# Function to create and save random line plots
def generate_random_line_plots(num_plots):
    for i in range(num_plots):

        # Generate random slope (m) and intercept (b)
        m = np.random.uniform(-7, 7)  # Random slope
        b = np.random.uniform(-5, 5)   # Random intercept
        print(m)

        # Generate x values from -5 to 5
        x_values = np.linspace(-5, 5, 100)
        # Calculate corresponding y values using the linear equation
        y_values = m * x_values + b

        # Create the plot
        plt.figure(figsize=(8, 8))
        plt.plot(x_values, y_values, color='b', linestyle='-')

        # Set the limits of the plot
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)

        plt.xticks([])
        plt.yticks([])

        # Add horizontal and vertical lines at zero
        plt.axhline(0, color='black', linewidth=0.5, ls='solid')
        plt.axvline(0, color='black', linewidth=0.5, ls='solid')

        # Save the plot as a JPG image
        plt.savefig(f'random_line_function_{i + 1}.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)

        # Close the plot to free memory
        plt.close()

# Input: Number of plots to generate
#num_plots = int(input("Enter the number of random line plots to generate: "))

def image_to_matrix(image_path, new_size=(200, 200)):
    # Load the image using imageio
    img = imageio.imread(image_path)
    
    # Get original dimensions
    original_height, original_width = img.shape[:2]

    # Create an empty array for the resized image
    img_resized = np.zeros((new_size[0], new_size[1], img.shape[2]), dtype=img.dtype)

    # Calculate the scaling factors
    height_scale = original_height / new_size[0]
    width_scale = original_width / new_size[1]

    for i in range(new_size[0]):
        for j in range(new_size[1]):
            # Find corresponding pixel in the original image
            orig_x = int(j * width_scale)
            orig_y = int(i * height_scale)
            img_resized[i, j] = img[orig_y, orig_x]

    return img_resized

# Example usage
num_plots = 1
generate_random_line_plots(num_plots)
image_path = 'random_line_function_1.jpg'  # Replace with your image file name
matrix = image_to_matrix(image_path, new_size=(200, 200)) 
print("Resized Matrix shape:", matrix.shape)
D.append(matrix)
print(matrix)
plt.figure()
plt.imshow(matrix, interpolation='nearest')
plt.savefig(f'image.jpg', format='jpg')

