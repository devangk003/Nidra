import os
import numpy as np
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_npy_shapes_and_integrity(npy_folder, label_folder):
    """Check the shapes of Numpy arrays and validate their integrity, including segment-label consistency."""
    npy_files = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.npy')]

    for npy_file in npy_files:
        npy_path = os.path.join(npy_folder, npy_file)
        label_path = os.path.join(label_folder, npy_file)  # Assuming labels have the same filename

        try:
            # Load mel spectrogram data
            data = np.load(npy_path)
            logging.info(f"Shape of {npy_path}: {data.shape}")

            # Check for deformities in the mel spectrogram array
            if np.isnan(data).any():
                logging.warning(f"Deformity found in {npy_path}: Contains NaN values.")
            if np.isinf(data).any():
                logging.warning(f"Deformity found in {npy_path}: Contains infinite values.")
            if data.size == 0:
                logging.warning(f"Deformity found in {npy_path}: Array is empty.")

            # Check if corresponding label file exists
            if not os.path.exists(label_path):
                logging.error(f"Missing corresponding label file for {npy_file}")
                continue

            # Load label data
            labels = np.load(label_path)
            logging.info(f"Shape of {label_path}: {labels.shape}")

            # Validate segment-label consistency
            if data.shape[0] != labels.shape[0]:
                logging.error(f"Mismatch between segments and labels in {npy_file}: "
                              f"{data.shape[0]} segments vs {labels.shape[0]} labels.")
        except Exception as e:
            logging.error(f"Error processing {npy_file}: {e}")

def main():
    mel_spectrogram_folder = "exports/mel_spectrograms"
    label_folder = "exports/labels"
    check_npy_shapes_and_integrity(mel_spectrogram_folder, label_folder)

if __name__ == "__main__":
    main()