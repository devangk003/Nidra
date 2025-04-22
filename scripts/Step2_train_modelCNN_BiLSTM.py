import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
import sys
from tqdm import tqdm
import gc
import signal
import atexit
import tempfile
from sklearn.utils import class_weight
import math

# Optimized cleanup function
def cleanup_tensorflow_resources():
    """Clean up TensorFlow resources gracefully."""
    try:
        # Clear any cached tensors and force garbage collection
        gc.collect()
        
        # Clear TensorFlow session
        tf.keras.backend.clear_session()
        
        # Reset GPU memory if available
        if tf.config.list_physical_devices('GPU'):
            for device in tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.reset_memory_stats(device)
                except:
                    pass
    except Exception as e:
        logging.warning(f"Error during cleanup: {e}")

# Register cleanup function at exit
atexit.register(cleanup_tensorflow_resources)

# Handle SIGINT gracefully (Ctrl+C)
def handle_sigint(sig, frame):
    logging.info("Received SIGINT. Cleaning up resources...")
    cleanup_tensorflow_resources()
    sys.exit(0)
    
signal.signal(signal.SIGINT, handle_sigint)

# Configure GPU memory growth to prevent OOM errors
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if (physical_devices):
        logging.info(f"Found {len(physical_devices)} GPU(s)")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            logging.info(f"Enabled memory growth for {device}")
    else:
        logging.warning("No GPU found. Using CPU for training (will be slower)")
except Exception as e:
    logging.warning(f"Error configuring GPU: {e}")

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MEL_SPECTROGRAM_DIR = 'exports/mel_spectrograms'
LABEL_DIR = 'exports/labels'
BATCH_SIZE = 8  # Reduced for memory efficiency
LEARNING_RATE = 0.0005  # Reduced for temporal model
EPOCHS = 2  # Increased for better temporal learning
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 4
CHECKPOINT_DIR = 'models/temporal_checkpoints'
CONTEXT_SIZE = 7  # Number of consecutive spectrograms to include (temporal context)
USE_TEMPORAL = True  # Flag to switch between temporal and non-temporal models

# Sleep stage constants (for rule-based processing)
WAKE = 0
REM = 1
LIGHT = 2
DEEP = 3

class MelSpectrogramGenerator:
    """Memory-efficient data generator for mel spectrograms."""
    
    def __init__(self, mel_files, label_files, batch_size=8, is_training=True):
        self.mel_files = mel_files
        self.label_files = label_files
        self.batch_size = batch_size
        self.is_training = is_training
        self.unused_files = list(range(len(mel_files)))  # Track unused files
        self.total_segments = self._count_total_segments()
        self.processed_files = set()
        
    def _count_total_segments(self):
        """Count total segments across all files."""
        total = 0
        for mel_file in self.mel_files:
            try:
                shape = np.load(mel_file, mmap_mode='r').shape
                total += shape[0]
            except Exception as e:
                logging.warning(f"Error counting segments in file {mel_file}: {e}")
                continue
        return total
                
    def generate(self):
        """Generator function that ensures all files are processed in each epoch."""
        while True:
            if not self.unused_files:
                # Reset unused files when all have been used
                self.unused_files = list(range(len(self.mel_files)))
                if self.is_training:
                    np.random.shuffle(self.unused_files)
            
            batch_x, batch_y = [], []
            
            for idx in self.unused_files[:]:
                try:
                    mel_file = self.mel_files[idx]
                    label_file = self.label_files[idx]
                    
                    # Log the file being processed
                    logging.info(f"Processing file: {mel_file} and {label_file}")
                    
                    mel_spec = np.load(mel_file, mmap_mode='r')
                    label = np.load(label_file, mmap_mode='r')
                    
                    segments = np.arange(len(mel_spec))
                    if self.is_training:
                        np.random.shuffle(segments)
                        
                    for seg_idx in segments:
                        spec = mel_spec[seg_idx].copy()
                        lbl = label[seg_idx].copy()
                        
                        # Make sure spec is 3-dimensional (add channel dimension if needed)
                        if spec.ndim == 2:
                            spec = np.expand_dims(spec, axis=-1)
                            # If grayscale, replicate to 3 channels
                            if spec.shape[-1] == 1:
                                spec = np.repeat(spec, 3, axis=-1)
                        
                        # Normalize spectrogram data
                        spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
                        
                        resized_spec = tf.image.resize(spec, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
                        
                        batch_x.append(resized_spec)
                        batch_y.append(lbl)
                        
                        if len(batch_x) >= self.batch_size:
                            yield np.array(batch_x), np.array(batch_y)
                            batch_x, batch_y = [], []
                    
                    # Mark file as used
                    self.unused_files.remove(idx)
                    
                except Exception as e:
                    logging.warning(f"Error processing file in generator: {e}")
                    self.unused_files.remove(idx)  # Skip problematic file
                    continue
            
            if len(batch_x) > 0:
                yield np.array(batch_x), np.array(batch_y)

    def generate_once(self):
        """Generate data for one pass through the dataset."""
        for idx in range(len(self.mel_files)):
            try:
                mel_file = self.mel_files[idx]
                label_file = self.label_files[idx]
                
                # Log the file being processed
                logging.info(f"Processing file: {mel_file} and {label_file}")
                
                mel_spec = np.load(mel_file, mmap_mode='r')
                label = np.load(label_file, mmap_mode='r')
                
                segments = np.arange(len(mel_spec))
                if self.is_training:
                    np.random.shuffle(segments)
                    
                batch_x, batch_y = [], []
                
                for seg_idx in segments:
                    spec = mel_spec[seg_idx].copy()
                    lbl = label[seg_idx].copy()
                    
                    # Make sure spec is 3-dimensional
                    if spec.ndim == 2:
                        spec = np.expand_dims(spec, axis=-1)
                        if spec.shape[-1] == 1:
                            spec = np.repeat(spec, 3, axis=-1)
                    
                    # Normalize spectrogram data
                    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
                    
                    resized_spec = tf.image.resize(spec, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
                    
                    batch_x.append(resized_spec)
                    batch_y.append(lbl)
                    
                    if len(batch_x) >= self.batch_size:
                        yield np.array(batch_x), np.array(batch_y)
                        batch_x, batch_y = [], []
                        
                if len(batch_x) > 0:
                    yield np.array(batch_x), np.array(batch_y)
                    
            except Exception as e:
                logging.warning(f"Error processing file in generator: {e}")
                continue


class TemporalSequenceGenerator(MelSpectrogramGenerator):
    """Generator that creates sequences of consecutive spectrograms for temporal processing."""
    
    def __init__(self, mel_files, label_files, batch_size=8, context_size=7, is_training=True):
        """
        Initialize with temporal context parameters.
        
        Args:
            mel_files: List of mel spectrogram files
            label_files: List of label files
            batch_size: Batch size for training
            context_size: Number of consecutive spectrograms to consider (should be odd)
            is_training: Whether this generator is for training
        """
        super().__init__(mel_files, label_files, batch_size, is_training)
        self.context_size = context_size
        # Modified total segments calculation for temporal sequences
        self.total_segments = self._count_total_segments_for_sequences()
        logging.info(f"Initialized temporal generator with context size {context_size}")
        
    def _count_total_segments_for_sequences(self):
        """Count segments available for sequence generation."""
        total = 0
        for mel_file in self.mel_files:
            try:
                shape = np.load(mel_file, mmap_mode='r').shape
                # Need at least context_size segments to create a sequence
                if shape[0] >= self.context_size:
                    # Each file can generate (n - context_size + 1) complete sequences
                    total += max(0, shape[0] - self.context_size + 1)
            except Exception as e:
                logging.warning(f"Error counting sequence segments in {mel_file}: {e}")
                continue
        return total
    
    def generate(self):
        """Generator function that creates batches of temporal sequences."""
        while True:
            if not self.unused_files:
                # Reset unused files when all have been used
                self.unused_files = list(range(len(self.mel_files)))
                if self.is_training:
                    np.random.shuffle(self.unused_files)
            
            batch_sequences = []  # Will hold sequences of shape (context_size, h, w, c)
            batch_labels = []     # Will hold labels for the center position of each sequence
            
            for idx in self.unused_files[:]:
                try:
                    mel_file = self.mel_files[idx]
                    label_file = self.label_files[idx]
                    
                    logging.info(f"Processing file for sequences: {mel_file}")
                    
                    # Load the entire file
                    mel_spec = np.load(mel_file, mmap_mode='r')
                    label = np.load(label_file, mmap_mode='r')
                    
                    # Skip if file is too short for a complete sequence
                    if len(mel_spec) < self.context_size:
                        logging.warning(f"File {mel_file} has fewer segments ({len(mel_spec)}) " 
                                       f"than context size ({self.context_size}). Skipping.")
                        self.unused_files.remove(idx)
                        continue
                    
                    # Calculate center offset for context window
                    half_context = self.context_size // 2
                    
                    # Generate start positions for sequences (maintaining original order)
                    # For training: create sequences starting at each position
                    # For validation: maintain temporal order 
                    start_positions = np.arange(len(mel_spec) - self.context_size + 1)
                    if self.is_training:
                        # Shuffle start positions but maintain sequence structure
                        np.random.shuffle(start_positions)
                    
                    for start_pos in start_positions:
                        # Extract the sequence and corresponding label
                        end_pos = start_pos + self.context_size
                        
                        # Get the sequence of spectrograms
                        sequence = []
                        for seq_idx in range(start_pos, end_pos):
                            # Get and preprocess each spectrogram in the sequence
                            spec = mel_spec[seq_idx].copy()
                            
                            # Handle dimensions
                            if spec.ndim == 2:
                                spec = np.expand_dims(spec, axis=-1)
                                if spec.shape[-1] == 1:
                                    spec = np.repeat(spec, 3, axis=-1)
                            
                            # Normalize
                            spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
                            
                            # Resize
                            resized_spec = tf.image.resize(spec, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
                            sequence.append(resized_spec.numpy())
                        
                        # Get the label for the center position
                        center_pos = start_pos + half_context
                        target_label = label[center_pos]
                        
                        batch_sequences.append(np.array(sequence))
                        batch_labels.append(target_label)
                        
                        # Yield batch when full
                        if len(batch_sequences) >= self.batch_size:
                            yield np.array(batch_sequences), np.array(batch_labels)
                            batch_sequences, batch_labels = [], []
                    
                    # Mark file as used
                    self.unused_files.remove(idx)
                    
                except Exception as e:
                    logging.warning(f"Error processing file for sequences: {e}")
                    import traceback
                    traceback.print_exc()
                    self.unused_files.remove(idx)
                    continue
            
            # Yield remaining sequences if any
            if batch_sequences:
                yield np.array(batch_sequences), np.array(batch_labels)


class SumLayer(tf.keras.layers.Layer):
    """Custom layer to replace Lambda for better serialization."""
    
    def __init__(self, axis=1):
        super(SumLayer, self).__init__()
        self.axis = axis
        
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)
        
    def get_config(self):
        config = super(SumLayer, self).get_config()
        config.update({"axis": self.axis})
        return config


def verify_data_structure():
    """Verify data structure and alignment for temporal sequence processing."""
    logging.info("Verifying data structure for temporal processing...")
    
    mel_files = [os.path.join(MEL_SPECTROGRAM_DIR, f) for f in os.listdir(MEL_SPECTROGRAM_DIR) 
                if f.endswith('.npy')]
    label_files = [os.path.join(LABEL_DIR, f) for f in os.listdir(LABEL_DIR) 
                  if f.endswith('.npy')]
    
    if not mel_files or not label_files:
        logging.error("No data files found.")
        return False
    
    # Check file correspondence
    mel_basenames = [os.path.basename(f) for f in mel_files]
    label_basenames = [os.path.basename(f) for f in label_files]
    
    if set(mel_basenames) != set(label_basenames):
        logging.error("Mismatching filenames between spectrograms and labels")
        return False
    
    # Check structure of a few files
    valid_files = 0
    sample_files = np.random.choice(mel_files, min(5, len(mel_files)), replace=False)
    
    for mel_file in sample_files:
        basename = os.path.basename(mel_file)
        label_file = os.path.join(LABEL_DIR, basename)
        
        try:
            mel_data = np.load(mel_file)
            label_data = np.load(label_file)
            
            # Check shapes
            if len(mel_data.shape) != 4:  # (num_segments, height, width, channels) or (num_segments, height, width) 
                logging.warning(f"Invalid spectrogram shape: {mel_data.shape} in {basename}")
                continue
                
            if mel_data.shape[0] != label_data.shape[0]:
                logging.warning(f"Mismatched dimensions: {mel_data.shape[0]} spectrograms vs {label_data.shape[0]} labels in {basename}")
                continue
            
            # Validate labels
            if len(label_data.shape) != 2 or label_data.shape[1] != NUM_CLASSES:
                logging.warning(f"Labels in {basename} not in one-hot format: {label_data.shape}")
                continue
                
            # Check if sum of one-hot encoding equals 1
            if not np.allclose(np.sum(label_data, axis=1), 1.0):
                logging.warning(f"Labels in {basename} not properly one-hot encoded")
                continue
                
            valid_files += 1
            duration_hours = mel_data.shape[0] * 30 / 3600  # Assuming 30-second epochs
            logging.info(f"✓ {basename}: {mel_data.shape[0]} epochs ({duration_hours:.2f} hours) - Valid")
            
        except Exception as e:
            logging.error(f"Error processing {basename}: {e}")
    
    if valid_files == 0:
        logging.error("No valid data files found for temporal processing")
        return False
    
    logging.info(f"Data verification complete: {valid_files}/{len(sample_files)} sampled files are valid")
    return True


def check_prerequisites():
    """Validate all prerequisites before starting training"""
    try:
        # Check if directories exist
        if not os.path.exists(MEL_SPECTROGRAM_DIR):
            raise FileNotFoundError(f"Mel spectrogram directory not found: {MEL_SPECTROGRAM_DIR}")
        if not os.path.exists(LABEL_DIR):
            raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")
        
        # Check if files exist
        mel_files = [f for f in os.listdir(MEL_SPECTROGRAM_DIR) if f.endswith('.npy')]
        label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.npy')]
        
        if not mel_files:
            raise FileNotFoundError(f"No mel spectrogram files (.npy) found in {MEL_SPECTROGRAM_DIR}")
        if not label_files:
            raise FileNotFoundError(f"No label files (.npy) found in {LABEL_DIR}")
            
        # Ensure files correspond
        if set([os.path.basename(f) for f in mel_files]) != set([os.path.basename(f) for f in label_files]):
            raise ValueError("Mismatch between mel spectrogram files and label files")
        
        # Verify data structure for temporal processing
        if USE_TEMPORAL and not verify_data_structure():
            logging.warning("Data structure verification for temporal processing failed")
            # Continue anyway but warn user
            
        # Create directories
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        return True
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Prerequisite check failed: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during prerequisite check: {e}")
        return False


def build_standard_model():
    """Build the standard non-temporal model (original implementation)."""
    logging.info("Building standard non-temporal model...")
    
    try:
        # Load EfficientNetB0 with pre-trained weights
        try:
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=INPUT_SHAPE
            )
        except tf.errors.ResourceExhaustedError:
            logging.error("GPU memory exhausted while loading EfficientNetB0.")
            logging.info("Attempting to continue with limited GPU memory...")
            # Try without preloaded weights
            base_model = EfficientNetB0(
                weights=None,
                include_top=False,
                input_shape=INPUT_SHAPE
            )
        
        # Freeze the base model
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        
        return model
    
    except tf.errors.ResourceExhaustedError:
        logging.error("GPU memory exhausted. Try using a smaller model or batch size.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error building model: {e}")
        sys.exit(1)


def build_temporal_model():
    """Build a model that processes sequences of spectrograms with temporal context."""
    logging.info(f"Building temporal model with context size {CONTEXT_SIZE}...")
    
    try:
        # Input shape includes the sequence dimension
        seq_input_shape = (CONTEXT_SIZE,) + INPUT_SHAPE
        
        # Create input layer
        inputs = tf.keras.Input(shape=seq_input_shape)
        
        # Load EfficientNetB0 for feature extraction (without top layers)
        try:
            cnn = EfficientNetB0(
                weights='imagenet',
                include_top=False, 
                pooling='avg',
                input_shape=INPUT_SHAPE
            )
        except tf.errors.ResourceExhaustedError:
            logging.warning("GPU memory exhausted loading EfficientNetB0 with weights.")
            logging.info("Falling back to model without pre-trained weights...")
            cnn = EfficientNetB0(
                weights=None, 
                include_top=False, 
                pooling='avg',
                input_shape=INPUT_SHAPE
            )
        
        # Freeze CNN weights
        cnn.trainable = False
        
        # Apply CNN to each spectrogram in the sequence
        # TimeDistributed wrapper applies the same layer to each time step
        x = layers.TimeDistributed(cnn)(inputs)  # Output: (batch, seq_len, cnn_features)
        
        # Add batch normalization for better training stability
        x = layers.BatchNormalization()(x)
        
        # Bidirectional LSTM to capture temporal patterns
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True)
        )(x)
        
        # Self-attention mechanism
        # We'll use a simple attention implementation
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Reshape((CONTEXT_SIZE,))(attention)  
        attention_weights = layers.Activation('softmax')(attention)
        attention_weights = layers.Reshape((CONTEXT_SIZE, 1))(attention_weights)
        
        # Multiply features by attention weights and sum
        context_vector = layers.Multiply()([x, attention_weights])
        context_vector = SumLayer(axis=1)(context_vector)
        
        # Final classification layers
        x = layers.Dense(128, activation='gelu')(context_vector)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        return model
        
    except Exception as e:
        logging.error(f"Error building temporal model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def apply_sleep_rules(predictions):
    """
    Apply sleep medicine rules to improve predictions.
    
    Args:
        predictions: Array of shape (num_epochs, num_classes) with softmax probabilities
        
    Returns:
        Array of shape (num_epochs, num_classes) with improved predictions
    """
    # Convert to class indices
    pred_classes = np.argmax(predictions, axis=1)
    
    # Create a copy of the predictions to avoid modifying the original
    smoothed = pred_classes.copy()
    
    # Rule 1: Minimum duration for sleep stages (in epochs)
    min_duration = {
        WAKE: 2,    # 1 minute minimum
        REM: 3,     # 1.5 minutes minimum
        LIGHT: 2,   # 1 minute minimum  
        DEEP: 4     # 2 minutes minimum
    }
    
    # Apply minimum duration rule by removing brief segments
    i = 0
    while i < len(smoothed):
        # Find end of current segment
        current_stage = smoothed[i]
        j = i + 1
        while j < len(smoothed) and smoothed[j] == current_stage:
            j += 1
            
        # Check if segment is too short
        segment_length = j - i
        if segment_length < min_duration.get(current_stage, 1):
            # Too short - look at surrounding stages for replacement
            if i > 0 and j < len(smoothed):
                # Use majority vote from neighboring stages
                previous_stage = smoothed[i-1]
                next_stage = smoothed[j]
                
                if previous_stage == next_stage:
                    # Same stage on both sides, use that
                    smoothed[i:j] = previous_stage
                else:
                    # Different stages, use probabilities to decide
                    avg_prev_conf = np.mean([predictions[idx, previous_stage] for idx in range(i, j)])
                    avg_next_conf = np.mean([predictions[idx, next_stage] for idx in range(i, j)])
                    replacement = previous_stage if avg_prev_conf > avg_next_conf else next_stage
                    smoothed[i:j] = replacement
            elif i > 0:
                # At the end, extend previous stage
                smoothed[i:j] = smoothed[i-1]
            elif j < len(smoothed):
                # At the beginning, extend next stage
                smoothed[i:j] = smoothed[j]
        
        i = j
    
    # Rule 2: Prevent physiologically implausible transitions
    for i in range(1, len(smoothed)):
        # REM to Deep Sleep is not plausible (needs Light Sleep transition)
        if smoothed[i] == DEEP and smoothed[i-1] == REM:
            smoothed[i] = LIGHT  # Use Light Sleep as intermediate
        
        # Deep Sleep to REM is also uncommon without Light Sleep
        if smoothed[i] == REM and smoothed[i-1] == DEEP:
            smoothed[i] = LIGHT  # Use Light Sleep as intermediate
        
        # Wake to Deep Sleep is uncommon (usually through Light Sleep)
        if smoothed[i] == DEEP and smoothed[i-1] == WAKE:
            smoothed[i] = LIGHT
    
    # Rule 3: Smooth isolated sleep stages (surrounded by same stage)
    for i in range(1, len(smoothed)-1):
        if smoothed[i-1] == smoothed[i+1] and smoothed[i] != smoothed[i-1]:
            # Isolated different stage
            if i >= 2 and i <= len(smoothed)-3:  # Check wider context
                # Check if it's truly isolated in a wider context
                if smoothed[i-2] == smoothed[i-1] and smoothed[i+1] == smoothed[i+2]:
                    smoothed[i] = smoothed[i-1]
    
    # Convert back to one-hot encoding
    smoothed_predictions = np.zeros_like(predictions)
    for i, cls in enumerate(smoothed):
        smoothed_predictions[i, cls] = 1.0
    
    return smoothed_predictions


def build_model():
    """Build appropriate model based on configuration."""
    if (USE_TEMPORAL):
        return build_temporal_model()
    else:
        return build_standard_model()


def save_model_summary(model, filepath='model_summary.txt'):
    """Save model summary to a file."""
    # Redirect stdout to a string buffer
    from io import StringIO
    import sys
    
    old_stdout = sys.stdout
    buffer = StringIO()
    sys.stdout = buffer
    
    # Print model summary
    model.summary()
    
    # Restore stdout
    sys.stdout = old_stdout
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(buffer.getvalue())
    
    logging.info(f"Model summary saved to {filepath}")


def ensure_serializable(obj):
    """Convert tensors to numpy arrays to make them serializable."""
    if isinstance(obj, tf.Tensor):
        return obj.numpy()
    elif isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_serializable(x) for x in obj]
    else:
        return obj


def main():
    """Main function with error handling and data generator approach."""
    try:
        # Validate prerequisites
        if not check_prerequisites():
            sys.exit(1)
        
        # Get file paths instead of loading all data
        mel_files = [os.path.join(MEL_SPECTROGRAM_DIR, f) for f in os.listdir(MEL_SPECTROGRAM_DIR) 
                    if f.endswith('.npy')]
        label_files = [os.path.join(LABEL_DIR, f) for f in os.listdir(LABEL_DIR) 
                      if f.endswith('.npy')]
        
        # Ensure matching filenames
        mel_basenames = [os.path.basename(f) for f in mel_files]
        label_basenames = [os.path.basename(f) for f in label_files]
        
        # Create paired lists of files
        paired_files = []
        for i, mel_file in enumerate(mel_files):
            basename = os.path.basename(mel_file)
            if basename in label_basenames:
                label_idx = label_basenames.index(basename)
                paired_files.append((mel_file, label_files[label_idx]))
        
        if len(paired_files) == 0:
            raise ValueError("No matching mel spectrogram and label files found")
        
        # Split files for train/validation
        np.random.shuffle(paired_files)
        split_idx = int(len(paired_files) * 0.8)  # 80% training, 20% validation
        train_files = paired_files[:split_idx]
        val_files = paired_files[split_idx:]
        
        train_mel_files = [t[0] for t in train_files]
        train_label_files = [t[1] for t in train_files]
        val_mel_files = [v[0] for v in val_files]
        val_label_files = [v[1] for v in val_files]
        
        # Create appropriate generators based on configuration
        if USE_TEMPORAL:
            logging.info("Creating temporal sequence generators...")
            train_generator = TemporalSequenceGenerator(
                train_mel_files, train_label_files, 
                batch_size=BATCH_SIZE, context_size=CONTEXT_SIZE, is_training=True)
            
            val_generator = TemporalSequenceGenerator(
                val_mel_files, val_label_files, 
                batch_size=BATCH_SIZE, context_size=CONTEXT_SIZE, is_training=False)
            
            logging.info(f"Using temporal model with context size {CONTEXT_SIZE}")
        else:
            logging.info("Creating standard spectrogram generators...")
            train_generator = MelSpectrogramGenerator(
                train_mel_files, train_label_files, batch_size=BATCH_SIZE, is_training=True)
            
            val_generator = MelSpectrogramGenerator(
                val_mel_files, val_label_files, batch_size=BATCH_SIZE, is_training=False)
            
            logging.info("Using standard non-temporal model")
        
        # Calculate steps per epoch
        steps_per_epoch = math.ceil(train_generator.total_segments / BATCH_SIZE)
        validation_steps = math.ceil(val_generator.total_segments / BATCH_SIZE)
        
        if steps_per_epoch == 0 or validation_steps == 0:
            raise ValueError("Not enough data for training with current batch size and context window")
            
        logging.info(f"Training with {train_generator.total_segments} samples in {len(train_files)} files")
        logging.info(f"Validating with {val_generator.total_segments} samples in {len(val_files)} files")
        logging.info(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")
        
        # Build appropriate model
        model = build_model()
        
        # Save model architecture summary
        save_model_summary(model, os.path.join(CHECKPOINT_DIR, 'model_summary.txt'))
        
        # Set up callbacks
        callbacks = [
            # Modified to save weights only
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(CHECKPOINT_DIR, 'best_model_weights.h5'),
                save_best_only=True,
                save_weights_only=True,  # <-- Add this parameter
                monitor='val_accuracy',
                mode='max'
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=5,
                monitor='val_accuracy',
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Add this code before model.fit in the main function

        # Calculate class weights properly with all classes represented
        try:
            # Get a representative sample of labels
            logging.info("Calculating class weights for balanced training...")
            sample_labels = []
            
            # Sample from training generator
            for i, (_, y_batch) in enumerate(train_generator.generate()):
                # Add labels from this batch
                batch_labels = np.argmax(y_batch, axis=1)
                sample_labels.extend(batch_labels)
                
                # Stop after collecting enough samples
                if len(sample_labels) >= 5000:  # Sample size large enough to be representative
                    break
            
            # Convert to numpy array
            sample_labels = np.array(sample_labels)
            
            # Compute class weights
            weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(sample_labels),
                y=sample_labels
            )
            
            # Create dictionary with weights for ALL possible classes (even if not in sample)
            class_weights = {}
            for i in range(NUM_CLASSES):
                # If class exists in sample, use computed weight, otherwise use 1.0
                idx = np.where(np.unique(sample_labels) == i)[0]
                if len(idx) > 0:
                    class_weights[i] = float(weights[idx[0]])
                else:
                    logging.warning(f"Class {i} not found in sample. Using default weight 1.0")
                    class_weights[i] = 1.0
            
            logging.info(f"Class weights: {class_weights}")

        except Exception as e:
            logging.warning(f"Error calculating class weights: {e}. Using equal weights.")
            class_weights = {i: 1.0 for i in range(NUM_CLASSES)}

        # Use class weights in model.fit
        history = model.fit(
            train_generator.generate(),
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=val_generator.generate(),
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        # Log a summary of the training history
        try:
            final_accuracy = history.history.get('accuracy', [0])[-1]
            final_val_accuracy = history.history.get('val_accuracy', [0])[-1]
            logging.info(f"Training completed. Final accuracy: {float(final_accuracy):.4f}, "
                         f"Validation accuracy: {float(final_val_accuracy):.4f}")
        except Exception as e:
            logging.warning(f"Could not log training history: {e}")
        
        # Save weights and full model
        try:
            # Save weights only (not full model)
            try:
                # Create directories if they don't exist
                model_dir = 'models/temporal' if USE_TEMPORAL else 'models/standard'
                os.makedirs(f'{model_dir}/weights', exist_ok=True)
                
                # Save weights
                weights_path = f'{model_dir}/weights/final_model_weights.h5'
                model.save_weights(weights_path)
                logging.info(f"Model weights saved as '{weights_path}'")
                
                # Save model configuration as JSON
                model_config = {
                    'context_size': CONTEXT_SIZE,
                    'input_shape': INPUT_SHAPE,
                    'num_classes': NUM_CLASSES,
                    'is_temporal': USE_TEMPORAL
                }
                
                import json
                with open(f'{model_dir}/model_config.json', 'w') as f:
                    json.dump(model_config, f)
                logging.info(f"Model configuration saved to {model_dir}/model_config.json")
                
                # Plot training history
                plt.figure(figsize=(12, 5))
                
                # Convert any tensor values to numpy for safe serialization
                history_dict = {}
                for key, value in history.history.items():
                    history_dict[key] = [float(v) for v in value]
                
                # Plot accuracy
                plt.subplot(1, 2, 1)
                plt.plot(history_dict['accuracy'], label='Train')
                plt.plot(history_dict['val_accuracy'], label='Validation')
                plt.title('Model Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                
                # Plot loss
                plt.subplot(1, 2, 2)
                plt.plot(history_dict['loss'], label='Train')
                plt.plot(history_dict['val_loss'], label='Validation')
                plt.title('Model Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                
                plt.tight_layout()
                history_plot_path = f'{model_dir}/training_history.png'
                plt.savefig(history_plot_path)
                plt.close()
                
                # Save history as JSON for future reference
                with open(f'{model_dir}/training_history.json', 'w') as f:
                    json.dump(history_dict, f)
                
                logging.info(f"Training history plot saved as '{history_plot_path}'")
                logging.info(f"Training history data saved to {model_dir}/training_history.json")
                
            except Exception as e:
                logging.error(f"Error saving model: {e}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            logging.error(f"Error saving model: {e}")
        
    except Exception as e:
        logging.error(f"Unexpected error in main process: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure proper cleanup of resources
        logging.info("Cleaning up resources...")
        if 'train_generator' in locals():
            del train_generator
        if 'val_generator' in locals():
            del val_generator
        gc.collect()
        tf.keras.backend.clear_session()

# Modified main execution
if __name__ == "__main__":
    main()
    # Ensure thorough cleanup at exit
    cleanup_tensorflow_resources()