import os

# Directories
MEL_SPECTROGRAM_DIR = 'exports/mel_spectrograms'
LABEL_DIR = 'exports/labels'

# Get filenames from both directories
mel_files = set([f for f in os.listdir(MEL_SPECTROGRAM_DIR) if f.endswith('.npy')])
label_files = set([f for f in os.listdir(LABEL_DIR) if f.endswith('.npy')])

print(f"Mel spectrogram files count: {len(mel_files)}")
print(f"Label files count: {len(label_files)}")

# Find differences
mel_only = mel_files - label_files
label_only = label_files - mel_files

if mel_only:
    print("\nFiles in mel_spectrograms but not in labels:")
    for f in sorted(mel_only):
        print(f"- {f}")

if label_only:
    print("\nFiles in labels but not in mel_spectrograms:")
    for f in sorted(label_only):
        print(f"- {f}")

if not mel_only and not label_only:
    print("\nFilenames match perfectly. Check for case sensitivity issues.")