import os
import librosa
import numpy as np
import xml.etree.myElementTree as ET
import soundfile as sf
from tqdm import tqdm
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Sleep stage mapping
SLEEP_STAGE_MAP = {
    "Wake": 0,
    "REM": 1,
    "NonREM1": 2,
    "NonREM2": 2,  # Combine NonREM1 and NonREM2 into a single class
    "NonREM3": 3
}

# Constants
SAMPLE_RATE = 16000
MEL_BANDS = 224  # Updated for model input
TIME_FRAMES = 224  # Updated for model input
EPOCH_DURATION = 30
HOP_LENGTH = 512
N_FFT = 2048

def load_audio(file_path):
    """Load an audio file and return waveform & sample rate."""
    logging.info(f"Loading audio file: {file_path}")
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(y) == 0:
            raise ValueError("Loaded audio signal is empty.")
        return y, sr
    except Exception as e:
        logging.error(f"Error loading audio file {file_path}: {e}")
        return None, None

def compute_mel_spectrogram(y, sr):
    """Convert waveform to a Mel spectrogram."""
    logging.info("Computing Mel spectrogram")
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=MEL_BANDS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = (mel_spec - np.min(mel_spec)) / (np.max(mel_spec) - np.min(mel_spec))
    return mel_spec

def segment_audio(y, sr, epoch_duration):
    """Segment audio into epochs and compute Mel spectrograms for each epoch."""
    logging.info("Segmenting audio into epochs")
    mel_spectrograms = []

    mel_spec = compute_mel_spectrogram(y, sr)
    
    num_segments = mel_spec.shape[1] // TIME_FRAMES
    for i in range(num_segments):
        start = i * TIME_FRAMES
        end = start + TIME_FRAMES
        mel_segment = mel_spec[:, start:end]
        
        if mel_segment.shape == (MEL_BANDS, TIME_FRAMES):
            mel_segment = np.expand_dims(mel_segment, axis=-1)
            mel_segment = np.repeat(mel_segment, 3, axis=-1)
            mel_spectrograms.append(mel_segment)
        else:
            logging.warning(f"Skipping segment {i} due to incorrect shape: {mel_segment.shape}")

    return np.array(mel_spectrograms)

def parse_rml(rml_path, segment_duration, num_segments):
    """Parse the RML file and generate labels for each segment."""
    logging.info(f"Parsing RML file: {rml_path}")
    try:
        tree = ET.parse(rml_path)
        root = tree.getroot()
        
        stages = []
        for stage in root.findall('.//Stage'):
            start_time = float(stage.get('Start'))
            stage_type = stage.get('Type')
            stages.append((start_time, SLEEP_STAGE_MAP[stage_type]))
        
        # Calculate end times for each stage
        for i in range(len(stages) - 1):
            stages[i] = (stages[i][0], stages[i + 1][0], stages[i][1])
        stages[-1] = (stages[-1][0], float('inf'), stages[-1][1])
        
        # Generate labels for each segment
        labels = np.zeros((num_segments, len(set(SLEEP_STAGE_MAP.values()))))
        for i in range(num_segments):
            segment_start_time = i * segment_duration
            segment_end_time = (i + 1) * segment_duration
            for start, end, stage in stages:
                if start <= segment_start_time < end or start < segment_end_time <= end:
                    labels[i, stage] = 1
                    break
        
        return labels
    except Exception as e:
        logging.error(f"Error parsing RML file {rml_path}: {e}")
        return None

def process_audio_file(audio_path, rml_path, mel_spectrogram_folder, label_folder):
    mel_filename = os.path.splitext(os.path.basename(audio_path))[0] + ".npy"
    mel_output_path = os.path.join(mel_spectrogram_folder, mel_filename)
    label_output_path = os.path.join(label_folder, mel_filename)

    if os.path.exists(mel_output_path) and os.path.exists(label_output_path):
        logging.info(f"Files already processed for {audio_path}. Skipping.")
        return True

    y, sr = load_audio(audio_path)
    if y is None or sr is None:
        return False

    mel_spectrograms = segment_audio(y, sr, EPOCH_DURATION)
    num_segments = len(mel_spectrograms)
    
    # Calculate segment duration based on spectrogram parameters
    samples_per_segment = TIME_FRAMES * HOP_LENGTH
    segment_duration = samples_per_segment / SAMPLE_RATE
    
    labels = parse_rml(rml_path, segment_duration, num_segments)
    if labels is None:
        return False

    min_length = min(len(mel_spectrograms), len(labels))
    mel_spectrograms = mel_spectrograms[:min_length]
    labels = labels[:min_length]

    logging.info(f"Mel spectrograms shape: {mel_spectrograms.shape}")
    logging.info(f"Labels shape: {labels.shape}")

    np.save(mel_output_path, mel_spectrograms)
    np.save(label_output_path, labels)
    logging.info(f"Saved files for {mel_filename}")

    return True

def process_patient(patient_folder, rml_folder, output_folder):
    mel_spectrogram_folder = os.path.join(output_folder, "mel_spectrograms")
    label_folder = os.path.join(output_folder, "labels")
    os.makedirs(mel_spectrogram_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    patient_no = os.path.basename(patient_folder)
    ambient_audio_file = os.path.join(patient_folder, f"{patient_no}_ambient.wav")
    tracheal_audio_file = os.path.join(patient_folder, f"{patient_no}_tracheal.wav")  # Kept for reference
    rml_file = os.path.join(rml_folder, f"{patient_no}.rml")

    logging.info(f"Processing patient {patient_no}")
    ambient_processed = process_audio_file(ambient_audio_file, rml_file, mel_spectrogram_folder, label_folder)
    # tracheal_processed = process_audio_file(tracheal_audio_file, rml_file, mel_spectrogram_folder, label_folder)

    if not ambient_processed:  # Changed condition to only check ambient
        logging.warning(f"No valid ambient audio file found for patient {patient_no}. Skipping patient.")

def main():
    audio_input_folder = "dataset/audio"
    rml_input_folder = "dataset/APNEA_RML"
    output_folder = "exports"

    os.makedirs(output_folder, exist_ok=True)

    patient_folders = [f for f in os.listdir(audio_input_folder) if os.path.isdir(os.path.join(audio_input_folder, f))]
    with tqdm(total=len(patient_folders), desc="Processing patients") as pbar:
        for patient_folder in patient_folders:
            patient_folder_path = os.path.join(audio_input_folder, patient_folder)
            rml_folder_path = os.path.join(rml_input_folder, patient_folder)
            if os.path.isdir(patient_folder_path) and os.path.isdir(rml_folder_path):
                process_patient(patient_folder_path, rml_folder_path, output_folder)
            pbar.update(1)

if __name__ == "__main__":
    main()