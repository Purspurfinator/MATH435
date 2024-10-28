# generateGraphs.py

This script generates synthetic graphs for different types of functions and saves them as both images and matrices. The matrices are saved in a single compressed `.npz` file for easy loading and training.

## Function Types

The script generates graphs for the following types of functions:
- Linear
- Quadratic
- Cubic
- Quartic
- Exponential
- Logarithmic
- Sine
- Cosine
- Tangent
- Absolute Value
- Square Root
- Reciprocal
- Piecewise

## How It Works

1. **Define Functions with Parameters**:
   - Each function type is defined with parameters that can be randomized.

2. **Add Random Noise to the x-values**:
   - Small random perturbations are added to the x-values to introduce variation.

3. **Generate Random Parameters for the Functions**:
   - Random parameters are generated for each function type to ensure variation within the same type of graph.

4. **Generate and Save the Graphs**:
   - The graphs are plotted, saved as images, converted to grayscale, resized, and stored in lists.
   - The images are saved for visual inspection, and the matrices are saved in a single compressed `.npz` file.

5. **Print Progress**:
   - The script outputs the progress of graph generation for each function type.

## Running the Script

1. **Save the Script**:
   - Ensure the script is saved as `generateGraphs.py`.

2. **Install Required Libraries**:
   - Ensure you have NumPy and Matplotlib installed. You can install them using:
     ```sh
     pip install numpy matplotlib
     ```

3. **Run the Script**:
   - Open a terminal or command prompt.
   - Navigate to the directory containing your script.
   - Run the script:
     ```sh
     python generateGraphs.py
     ```

## Output

- The script generates synthetic graphs and saves them in the `dataset/images` directory.
- The matrices representing the graphs are saved in a single compressed `.npz` file named `graphs_dataset.npz` in the `dataset` directory.

## Conclusion

This script provides a way to generate synthetic graphs for various types of functions, ensuring a diverse dataset for training machine learning models.