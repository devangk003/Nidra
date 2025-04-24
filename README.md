# Nidra: Sleep Stage Classification with Deep Learning
<p align="center">
  <img src="https://github.com/devangk003/Nidra/blob/main/Nidra-logo.png" width="550" alt="Nidra-Logo">
</p>


Nidra is an automated sleep stage classification system that uses deep learning to analyze EEG/PSG data and classify sleep into Wake, REM, Light Sleep, and Deep Sleep stages. The model employs a temporal CNN-LSTM architecture with attention mechanisms to incorporate contextual information from surrounding epochs.

## Features

- Processes EDF files to extract signals
- Converts signals to mel-spectrograms for visual pattern recognition
- Classifies sleep into four stages (Wake, REM, Light Sleep, Deep Sleep)
- Utilizes context from surrounding epochs for improved accuracy
- Generates visual hypnograms for sleep stage visualization

## Requirements

- Python 3.10
- TensorFlow 2.10
- NumPy, SciPy, Matplotlib
- PyEDFlib
- Librosa
- tqdm

Full requirements are available in `requirements.txt`

# Installation


### 1. Clone the repository
`git clone https://github.com/yourusername/nidra.git`
`cd nidra`

### 2. Create necessary directories (if they don't exist)
```bash
mkdir -p dataset/APNEA_EDF dataset/APNEA_RML dataset/audio
mkdir -p exports/mel_spectrograms exports/labels exports/hypnogram
mkdir -p models/temporal/weights models/temporal_checkpoints
mkdir -p models/standard/weights models/checkpoints
```

### 3. Install dependencies
`pip install -r requirements.txt`

## 🚧 Current Status

The model compilation step is currently pending due to a TensorFlow EagerTensor serialization issue. Saved model weights (`.h5` or `.pt`) have been uploaded for demonstration purposes.

The goal of this upload is to showcase project design, implementation strategy, and deep learning workflow to recruiters and collaborators.
I am actively working on resolving this issue and will update the repository once the model compilation is successful.

## Usage Guide
### Step 1: Convert EDF Files to WAV
- Convert EDF files containing sleep recordings to WAV format. Place your raw EDF files in APNEA_EDF.

`python scripts/PreStep_edf_to_wav.py`

This script extracts "Tracheal" and "Mic" audio channels and saves them as .wav files in the audio directory, maintaining the original folder structure.

### Step 2: Format RML Annotation Files

1. Manually Edit RML Files:Format sleep annotation files (RML) to a standardized XML structure.

Manually Edit RML Files: Place your RML files in APNEA_RML. Open each RML file and scroll down until you see the <StagingData StagingDisplay="NeuroAdultAASM"> section. Keep only the <UserStaging> block within it, like this example:
```bash
<StagingData StagingDisplay="NeuroAdultAASM">
         <UserStaging>
            <NeuroAdultAASMStaging>
               <Stage Type="Wake" Start="0" />
               .
               .
               .
               <Stage Type="NonREM2" Start="17760" />
               <Stage Type="Wake" Start="17880" />
            </NeuroAdultAASMStaging>
         </UserStaging>
```

Remove everything else outside this <StagingData> block (or ensure only this structure remains).

2. Run the Formatting Script:

`python scripts/PreStep_convert_rml_format.py [path/to/rml_directory]`
### Example: python scripts/PreStep_convert_rml_format.py dataset/APNEA_RML

This script normalizes XML formatting, flattens nested stage elements, converts start times to float, and adds indentation. It modifies the RML files in place.

### Step 3: Generate Mel Spectrograms and Labels
Process the extracted WAV audio files and formatted RML annotations into model-ready data (NumPy arrays).

`python scripts/Step1_convert_audio_to_mel_and_label.py`

This script converts audio to mel spectrograms, processes RML annotations into label arrays, aligns them, and saves the processed data as .npy files in mel_spectrograms and labels.

### Step 4: Train the Model
Train the sleep stage classifier using the generated spectrograms and labels.

`python scripts/Step2_train_modelCNN_BiLSTM.py`

Options:

--temporal: Use temporal model (default: True, controlled by USE_TEMPORAL flag in the script).
Training parameters like epochs, batch size, learning rate are set within the script.
Model weights and checkpoints are saved in the models directory.

### Step 5: Test Model Performance
Evaluate the trained model's performance on the dataset.

`python scripts/test_trained_model.py [--file FILENAME.npy]`

- Loads the trained model weights (final_model_weights.h5 or best_model_weights.h5).
- Processes test files (or a specific file if --file is used).
- Generates hypnogram visualizations (.png) in hypnogram.
- Creates confusion matrices (.png) in hypnogram.
- Outputs accuracy metrics to the console and a summary file (training_accuracy_summary.txt).

### Step 6: Make Predictions on New Audio
Use the trained model to predict sleep stages for a new audio recording. Place the new recording in a recordings directory (create it if needed).


`python scripts/Step3_Use_Model.py`
### The script will prompt for the sleep start time.

- Processes audio files from the recordings directory.
- Loads the trained model weights.
- Makes predictions and saves predicted labels (.npy) to labels.
- Generates a hypnogram visualization (.png) in hypnogram.

### Step 7: Fine-tune the Model (Optional)
Fine-tune the existing trained model with additional data. Ensure new data is processed into mel_spectrograms and labels.

`python scripts/Step4_fine_tune.py`

- Loads the pre-trained model (assumes ImageNet weights for the base).
- Applies data augmentation.
- Fine-tunes the model on the available data.
- Saves the fine-tuned model as fine_tuned_sleep_stage_model.h5.

### Step 8: Check Model Compilation (Utility)
Verify that the model architecture can be rebuilt and weights can be loaded correctly.

`python scripts/Check_compile_model.py`

- Loads configuration and weights.
- Builds the model architecture.
- Compiles the model.
- Runs a simple verification prediction.


# Model Architecture
- The primary model uses an EfficientNetB0 CNN base pre-trained on ImageNet for feature extraction from mel spectrograms.

- Temporal Model (USE_TEMPORAL = True):
    -  TimeDistributed CNN layer applied to each spectrogram in a sequence (context window, default size 7).
    - BatchNormalization.
    - Bidirectional LSTM layer to capture temporal dependencies.
    - Self-Attention mechanism to weigh the importance of different epochs in the context window.
    - Dense classification layers.
- Standard Model (USE_TEMPORAL = False):
    - GlobalAveragePooling2D after the CNN base.
    - Dense classification layers.

# Performance
On benchmark datasets, the model achieves approximately:

- Overall accuracy: ~65%
- Class-wise accuracies:
    - Wake: TBD
    - REM: TBD
    - Light Sleep: TBD
    - Deep Sleep: TBD


# Utility Scripts
Several diagnostic and utility scripts are available in the scripts/utils/ (or main scripts) directory for debugging and verification:

- diagnose_tensorflow_GPU.py: Verify TensorFlow GPU setup.
- diagnose_Single_audio_mel_and_rml.py: Test processing pipeline for a single file.
- diagnose_read_label_npy.py: Inspect contents of label .npy files.
- diagnose_filename_mismatch.py: Check for missing or extra files between spectrograms and labels.
- diagnose_check_npy_shapes.py: Validate shapes and integrity of .npy data files.
- Test_rml_to_hypno.py: Test conversion from RML to a hypnogram plot.


# Citation

If you use this code in your research, please consider citing it:

```bash
@software{nidra_2025,
  author = {[Devang Kumawat]},
  title = {Nidra},
  year = {2025},
  url = {https://github.com/devangk003/Nidra}
}
```

# License
This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

# Acknowledgments
Dataset Used: Georgia Korompili, Anastasia Amfilochiou, Lampros Kokkalas, et al. (2022) . PSG-Audio. V3. Science Data Bank. https://doi.org/10.11922/sciencedb.00345.


            
