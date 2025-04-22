import os
import pyedflib
import soundfile as sf
from tqdm import tqdm

def edf_to_wav():
    """Convert nested EDF files to WAV with original structure"""
    # Use raw strings for Windows paths
    edf_root = r"E:\Nidra\dataset\APNEA_EDF"
    audio_root = r"E:\Nidra\dataset\audio"

    for root, dirs, files in os.walk(edf_root):
        for edf_file in tqdm(files, desc="Processing EDF files"):
            if not edf_file.endswith(".edf"):
                continue
            
            # Create output directory using os.path.join
            patient_path = os.path.relpath(root, edf_root)
            patient_dir = os.path.join(audio_root, patient_path)
            os.makedirs(patient_dir, exist_ok=True)
            
            full_edf_path = os.path.join(root, edf_file)
            base_name = os.path.splitext(edf_file)[0]

            try:
                with pyedflib.EdfReader(full_edf_path) as reader:
                    channels = reader.getSignalLabels()
                    
                    if "Tracheal" in channels:
                        tracheal_idx = channels.index("Tracheal")
                        tracheal_fs = reader.getSampleFrequency(tracheal_idx)
                        tracheal_signal = reader.readSignal(tracheal_idx)
                        sf.write(
                            os.path.join(patient_dir, f"{base_name}_tracheal.wav"),
                            tracheal_signal,
                            int(tracheal_fs),
                            subtype="PCM_16"
                        )
                    else:
                        print(f"\nTracheal channel not found in {edf_file}")
                    
                    if "Mic" in channels:
                        mic_idx = channels.index("Mic")
                        mic_fs = reader.getSampleFrequency(mic_idx)
                        mic_signal = reader.readSignal(mic_idx)
                        sf.write(
                            os.path.join(patient_dir, f"{base_name}_ambient.wav"),
                            mic_signal,
                            int(mic_fs),
                            subtype="PCM_16"
                        )
                    else:
                        print(f"\nMic channel not found in {edf_file}")

            except Exception as e:
                print(f"\nError processing {edf_file}: {str(e)}")
                continue

if __name__ == "__main__":
    edf_to_wav()