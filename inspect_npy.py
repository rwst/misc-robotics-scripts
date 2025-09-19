import numpy as np
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Inspect a .npy file.')
parser.add_argument('filename', type=str, help='The path to the .npy file to inspect.')
args = parser.parse_args()

# Load the .npy file
data = np.load(args.filename)

# Print the array to the console
print(data)

# You can also check its properties
print("Shape:", data.shape)
print("Data type:", data.dtype)
print("Size (number of elements):", data.size)
