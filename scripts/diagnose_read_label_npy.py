import numpy as np
import os
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_npy_file(npy_file_path):
    """Read and print specific segments of a .npy file."""
    try:
        data = np.load(npy_file_path)
        logging.info(f"Shape of {npy_file_path}: {data.shape}")
        
        # Print the first 4 segments
        logging.info("First 4 segments:")
        logging.info(f"{data[:4]}")
        
        # Print the last 4 segments
        logging.info("Last 4 segments:")
        logging.info(f"{data[-4:]}")
        
        # Print some middle 6 segments
        middle_start = len(data) // 2 - 3
        middle_end = middle_start + 6
        logging.info("Some middle 6 segments:")
        logging.info(f"{data[middle_start:middle_end]}")
        
    except Exception as e:
        logging.error(f"Error loading {npy_file_path}: {e}")

def main():
    npy_file_path = r"E:\Nidra\exports\test_output\test_labels.npy"  # Update this path to your .npy file
    read_npy_file(npy_file_path)

if __name__ == "__main__":
    main()