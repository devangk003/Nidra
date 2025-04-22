import os
import librosa
import numpy as np
import xml.etree.myElementTree as ET
import matplotlib.pyplot as plt
import logging
from mypathlib import Path

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants from Step2_convert_audio_to_mel.py
SAMPLE_RATE = 16000
MEL_BANDS = 224
TIME_FRAMES = 224
EPOCH_DURATION = 30
HOP_LENGTH = 512
N_FFT = 2048

SLEEP_STAGE_MAP = {
    "Wake": 0,
    "REM": 1,
    "NonREM1": 2,
    "NonREM2": 2,
    "NonREM3": 3
}

def visualize_mel_spectrogram(mel_spec, index, output_dir):
    """Visualize a single mel spectrogram."""
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec[:,:,0], aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram - Segment {index}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'mel_spec_{index}.png'))
    plt.close()

def test_single_file():
    # Test paths
    audio_path = r"E:\Nidra\dataset\audio\00000995-100507\00000995-100507_ambient.wav"
    rml_path = r"E:\Nidra\dataset\APNEA_RML\00000995-100507\00000995-100507.rml"
    output_dir = "test_output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Import functions from main script
    from Step1_convert_audio_to_mel_and_label import (
        load_audio, 
        compute_mel_spectrogram, 
        segment_audio, 
        parse_rml
    )
    
    # Process audio
    y, sr = load_audio(audio_path)
    mel_spectrograms = segment_audio(y, sr, EPOCH_DURATION)
    
    # Calculate segment duration for labels
    samples_per_segment = TIME_FRAMES * HOP_LENGTH
    segment_duration = samples_per_segment / SAMPLE_RATE
    
    # Process labels
    labels = parse_rml(rml_path, segment_duration, len(mel_spectrograms))
    
    # Print shapes and stats
    logging.info(f"Mel spectrograms shape: {mel_spectrograms.shape}")
    logging.info(f"Labels shape: {labels.shape}")
    
    # Visualize first 3 spectrograms
    for i in range(min(3, len(mel_spectrograms))):
        visualize_mel_spectrogram(mel_spectrograms[i], i, output_dir)
    
    # Save test samples
    np.save(os.path.join(output_dir, "test_mel_specs.npy"), mel_spectrograms[:3])
    np.save(os.path.join(output_dir, "test_labels.npy"), labels[:3])

if __name__ == "__main__":
    test_single_file()