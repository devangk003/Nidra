import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from pydub import AudioSegment
import logging
import sys
from datetime import datetime, timedelta # Import datetime objects
import noisereduce as nr

# --- Add the script directory to sys.path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
# --- End of path addition ---

# Import necessary functions from the conversion script
try:
    from Step1_convert_audio_to_mel_and_label import load_audio, segment_audio, SAMPLE_RATE, EPOCH_DURATION
    from Step2_train_modelCNN_BiLSTM import build_model, INPUT_SHAPE, NUM_CLASSES # Import build_model and necessary constants
except ImportError as e:
    logging.error(f"Could not import from i2_convert_audio_to_mel.py. Error: {e}")
    sys.exit(1)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_to_wav(audio_path):
    # ... (function remains the same) ...
    try:
        file_name, file_extension = os.path.splitext(audio_path)
        wav_path = audio_path # Assume it's already wav initially

        if file_extension.lower() != '.wav':
            logging.info(f"Converting {audio_path} to WAV format...")
            audio = AudioSegment.from_file(audio_path)
            wav_path = file_name + '.wav'
            audio.export(wav_path, format='wav')
            logging.info(f"Converted to {wav_path}")
            return wav_path
        return audio_path # Return the original path if it's already .wav
    except Exception as e:
        logging.error(f"Error converting {audio_path} to WAV: {e}")
        return None # Return None if conversion fails


# --- Updated plot_hypnogram function ---
def plot_hypnogram(preds, save_path, start_time):
    """Plot the predicted hypnogram with sleep stages, starting at start_time."""
    label_map = {0: 'Wake', 1: 'REM', 2: 'Light Sleep', 3: 'Deep Sleep'}
    stage_order = {'Wake': 0, 'REM': 1, 'Light Sleep': 2, 'Deep Sleep': 3}

    mapped_preds_names = [label_map.get(pred, 'Wake') for pred in preds]
    numeric_plot_preds = [stage_order[stage] for stage in mapped_preds_names]

    plt.figure(figsize=(15, 4))
    plt.step(range(len(numeric_plot_preds)), [-p for p in numeric_plot_preds], where='post', label='Predicted')

    plt.yticks([0, -1, -2, -3], ['Wake', 'REM', 'Light Sleep', 'Deep Sleep'])
    plt.ylim(-3.5, 0.5)

    # --- X-axis Time Calculation ---
    total_epochs = len(preds)
    # Determine tick positions (every hour = 120 epochs if EPOCH_DURATION is 30s)
    epochs_per_hour = 3600 // EPOCH_DURATION
    tick_positions = np.arange(0, total_epochs + 1, epochs_per_hour)

    # Calculate tick labels based on start_time
    tick_labels = []
    for pos in tick_positions:
        # Calculate the time elapsed in seconds from the start
        time_elapsed_seconds = pos * EPOCH_DURATION
        # Add the elapsed time to the start_time
        current_time = start_time + timedelta(seconds=time_elapsed_seconds)
        # Format the time string
        tick_labels.append(current_time.strftime("%I:%M %p")) # Format: HH:MM AM/PM

    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right') # Rotate labels for better readability
    plt.xlim(0, total_epochs) # Ensure x-axis covers the entire duration
    # --- End of X-axis Time Calculation ---

    plt.xlabel('Time')
    plt.ylabel('Sleep Stage')
    plt.title('Predicted Hypnogram')
    plt.grid(True, axis='x', linestyle=':')
    plt.tight_layout() # Adjust layout to prevent labels overlapping

    try:
        plt.savefig(save_path)
        logging.info(f"Hypnogram saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving hypnogram to {save_path}: {e}")
    plt.close()
# --- End of updated plot_hypnogram function ---

def get_start_time_from_user():
    """Prompts the user for sleep start time and validates it."""
    while True:
        time_str = input("Please enter the sleep start time (e.g., 10:30 PM): ")
        try:
            # Attempt to parse the time string. Use a dummy date.
            # %I for 12-hour clock, %M for minute, %p for AM/PM
            start_time = datetime.strptime(time_str, "%I:%M %p")
            # We only care about the time part for plotting relative intervals,
            # but keeping it as datetime is convenient for timedelta.
            # You could replace the date part if needed:
            # start_time = start_time.replace(year=2000, month=1, day=1) # Example dummy date
            return start_time
        except ValueError:
            logging.error("Invalid time format. Please use HH:MM AM/PM (e.g., 10:30 PM or 09:15 AM).")

def main():
    # --- Get Start Time ---
    start_time = get_start_time_from_user()
    logging.info(f"Using sleep start time: {start_time.strftime('%I:%M %p')}")
    # --- End Get Start Time ---

    # Define paths
    audio_dir = 'recordings'
    model_dir = 'models/checkpoints'
    weights_filename = 'final_model_weights.h5'
    weights_path = os.path.join(model_dir, weights_filename)
    output_dir = os.path.join('exports', 'labels')
    hypnogram_dir = os.path.join('exports', 'hypnograms')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(hypnogram_dir, exist_ok=True)

    if not os.path.exists(weights_path):
        logging.error(f"Weights file not found at {weights_path}. Please ensure the trained weights exist.")
        sys.exit(1)

    # --- Build Model and Load Weights ---
    try:
        logging.info("Building model architecture...")
        model = build_model() # Call the function to create the architecture
        logging.info(f"Loading weights from {weights_path}...")
        model.load_weights(weights_path) # Load only the weights
        logging.info("Model built and weights loaded successfully.")
    except Exception as e:
        logging.error(f"Error building model or loading weights: {e}")
        sys.exit(1)
    # --- End Build Model and Load Weights ---

    try:
        audio_files_to_process = [f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]
        if not audio_files_to_process:
            logging.warning(f"No audio files found in {audio_dir}")
            return
    except FileNotFoundError:
        logging.error(f"Audio directory not found: {audio_dir}")
        return

    for audio_file in tqdm(audio_files_to_process, desc="Processing Audio Files"):
        # ... (rest of the loop remains the same until plotting) ...
        base_name = os.path.splitext(audio_file)[0]
        audio_path = os.path.join(audio_dir, audio_file)
        label_file_path = os.path.join(output_dir, f"{base_name}.npy")
        hypnogram_file_path = os.path.join(hypnogram_dir, f"{base_name}.png")

        if os.path.exists(label_file_path) and os.path.exists(hypnogram_file_path):
            logging.info(f"Skipping already processed file: {audio_file}")
            continue

        logging.info(f"Processing file: {audio_file}")
        wav_audio_path = convert_to_wav(audio_path)
        if wav_audio_path is None: continue
        y, sr = load_audio(wav_audio_path)
        if y is None:
            if wav_audio_path != audio_path: try: os.remove(wav_audio_path) except OSError: pass
            continue
        logging.info(f"Generating mel spectrograms for {audio_file}...")
        mel_specs = segment_audio(y, sr, EPOCH_DURATION)
        if mel_specs.shape[0] == 0:
            logging.warning(f"No valid mel spectrogram segments generated for {audio_file}. Skipping.")
            if wav_audio_path != audio_path: try: os.remove(wav_audio_path) except OSError: pass
            continue
        logging.info(f"Generated {mel_specs.shape[0]} segments with shape {mel_specs.shape[1:]}")
        logging.info("Predicting sleep stages...")
        try:
            preds = model.predict(mel_specs)
        except Exception as e:
            logging.error(f"Error during model prediction for {audio_file}: {e}")
            if wav_audio_path != audio_path: try: os.remove(wav_audio_path) except OSError: pass
            continue
        predicted_stages = np.argmax(preds, axis=1)
        try:
            np.save(label_file_path, predicted_stages)
            logging.info(f"Predicted labels saved to {label_file_path}")
        except Exception as e:
            logging.error(f"Error saving predicted labels to {label_file_path}: {e}")

        # --- Pass start_time to plot_hypnogram ---
        plot_hypnogram(predicted_stages, hypnogram_file_path, start_time)
        # --- End Pass start_time ---

        if wav_audio_path != audio_path:
            try:
                os.remove(wav_audio_path)
                logging.info(f"Removed temporary WAV file: {wav_audio_path}")
            except OSError as e:
                logging.warning(f"Could not remove temporary WAV file {wav_audio_path}: {e}")

if __name__ == "__main__":
    main()
