import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging
import sys
import gc
import seaborn as sns
import argparse

# Suppress verbose memory logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Limit TensorFlow GPU memory growth
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants - matching train_model.py
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 4
CONTEXT_SIZE = 7
USE_TEMPORAL = True  # Set to False if you trained a non-temporal model
BATCH_SIZE = 8
MEL_SPECTROGRAM_DIR = 'exports/mel_spectrograms'
LABEL_DIR = 'exports/labels'
HYPNOGRAM_DIR = 'exports/hypnogram'
SLEEP_STAGES = ['Wake', 'REM', 'Light Sleep', 'Deep Sleep']
COLORS = ['gold', 'tomato', 'skyblue', 'darkblue']  # Colors for the hypnogram stages

# Create hypnogram directory if it doesn't exist
os.makedirs(HYPNOGRAM_DIR, exist_ok=True)

# Custom layer definition from train_model.py
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

def cleanup_tensorflow_resources():
    """Clean up TensorFlow resources gracefully."""
    try:
        gc.collect()
        tf.keras.backend.clear_session()
        if tf.config.list_physical_devices('GPU'):
            for device in tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.reset_memory_stats(device)
                except:
                    pass
    except Exception as e:
        logging.warning(f"Error during cleanup: {e}")

def configure_gpu():
    """Configure GPU for optimal performance."""
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if (physical_devices):
            logging.info(f"Found {len(physical_devices)} GPU(s)")
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                logging.info(f"Enabled memory growth for {device}")
            return True
        else:
            logging.warning("No GPU found. Using CPU for inference")
            return False
    except Exception as e:
        logging.warning(f"Error configuring GPU: {e}")
        return False

def build_standard_model():
    """Build the standard non-temporal model (from train_model.py)."""
    logging.info("Building standard model...")
    
    try:
        base_model = EfficientNetB0(
            weights=None,  # We'll load our own weights
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
    except Exception as e:
        logging.error(f"Error building standard model: {e}")
        sys.exit(1)

def build_temporal_model():
    """Build a model that processes sequences of spectrograms with temporal context (from train_model.py)."""
    logging.info(f"Building temporal model with context size {CONTEXT_SIZE}...")
    
    try:
        # Input shape includes the sequence dimension
        seq_input_shape = (CONTEXT_SIZE,) + INPUT_SHAPE
        
        # Create input layer
        inputs = tf.keras.Input(shape=seq_input_shape)
        
        # Load EfficientNetB0 for feature extraction (without top layers)
        cnn = EfficientNetB0(
            weights=None,  # We'll load our own weights
            include_top=False, 
            pooling='avg',
            input_shape=INPUT_SHAPE
        )
        
        # Freeze CNN weights
        cnn.trainable = False
        
        # Apply CNN to each spectrogram in the sequence
        x = layers.TimeDistributed(cnn)(inputs)
        
        # Add batch normalization
        x = layers.BatchNormalization()(x)
        
        # Bidirectional LSTM
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True)
        )(x)
        
        # Self-attention mechanism
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

def load_model():
    """Load model with weights from the training script location."""
    try:
        # Define paths based on model type
        model_dir = 'models/temporal' if USE_TEMPORAL else 'models/standard'
        weights_path = f'{model_dir}/weights/final_model_weights.h5'
        
        # Check if weights exist, if not try checkpoint location
        if not os.path.exists(weights_path):
            checkpoint_dir = 'models/temporal_checkpoints' if USE_TEMPORAL else 'models/checkpoints'
            alt_weights_path = f'{checkpoint_dir}/best_model_weights.h5'
            if os.path.exists(alt_weights_path):
                weights_path = alt_weights_path
                logging.info(f"Using checkpoint weights at {weights_path}")
            else:
                raise FileNotFoundError(f"No model weights found at {weights_path} or {alt_weights_path}")
        else:
            logging.info(f"Using final weights at {weights_path}")
            
        # Build appropriate model based on configuration
        if USE_TEMPORAL:
            model = build_temporal_model()
        else:
            model = build_standard_model()
        
        # Load weights
        model.load_weights(weights_path)
        logging.info(f"Model weights loaded from {weights_path}")
        
        # Compile model with same configuration as training
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def preprocess_spectrogram(spec):
    """Apply exactly the same preprocessing as during training."""
    # Make a copy to avoid modifying the original
    spec = spec.copy()
    
    # Normalize to [0,1] range
    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
    
    # Make sure spec is 3-dimensional (add channel dimension if needed)
    if spec.ndim == 2:
        spec = np.expand_dims(spec, axis=-1)
        # If grayscale, replicate to 3 channels
        if spec.shape[-1] == 1:
            spec = np.repeat(spec, 3, axis=-1)
    
    # Resize to match the input shape
    resized_spec = tf.image.resize(spec, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
    
    return resized_spec.numpy()  # Convert to numpy for consistency

def get_training_files():
    """Get all available training files that have both spectrograms and labels."""
    mel_files = [f for f in os.listdir(MEL_SPECTROGRAM_DIR) if f.endswith('.npy')]
    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.npy')]
    
    # Find files that exist in both directories
    common_files = []
    for file in mel_files:
        if file in label_files:
            common_files.append(file)
    
    if not common_files:
        logging.error("No matching spectrogram and label files found")
        sys.exit(1)
    
    return common_files

def generate_hypnogram(predictions, true_labels, filename, detailed=True):
    """Generate clear hypnogram showing predictions for each epoch."""
    # Convert one-hot encoded labels to class indices
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1)
    
    epochs = range(len(pred_classes))
    
    # Create plot with two aligned subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot true sleep stages (top)
    ax1.step(epochs, true_classes, where='post', color='black', alpha=0.5)
    ax1.scatter(epochs, true_classes, c=[COLORS[i] for i in true_classes], s=50)
    ax1.set_title('True Sleep Stages', fontsize=14)
    ax1.set_yticks(range(NUM_CLASSES))
    ax1.set_yticklabels(SLEEP_STAGES, fontsize=12)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax1.set_ylim(-0.5, NUM_CLASSES - 0.5)
    
    # Plot predicted sleep stages (bottom)
    ax2.step(epochs, pred_classes, where='post', color='black', alpha=0.5)
    ax2.scatter(epochs, pred_classes, c=[COLORS[i] for i in pred_classes], s=50)
    ax2.set_title('Predicted Sleep Stages', fontsize=14)
    ax2.set_xlabel('Epoch Number', fontsize=12)
    ax2.set_yticks(range(NUM_CLASSES))
    ax2.set_yticklabels(SLEEP_STAGES, fontsize=12)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax2.set_ylim(-0.5, NUM_CLASSES - 0.5)
    
    # Add legend
    for i, stage in enumerate(SLEEP_STAGES):
        ax1.scatter([], [], c=COLORS[i], label=stage, s=50)
    fig.legend(fontsize=12, loc='upper right')
    
    # Add accuracy information
    accuracy = np.mean(pred_classes == true_classes)
    plt.figtext(0.5, 0.01, f'Accuracy: {accuracy:.4f}', ha='center', fontsize=14, 
                bbox={'facecolor':'lightgray', 'alpha':0.7, 'pad':8})
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, bottom=0.07)
    
    # Save plot
    output_path = os.path.join(HYPNOGRAM_DIR, f"{os.path.splitext(filename)[0]}_hypnogram.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    
    logging.info(f"Hypnogram saved to {output_path}")
    
    if detailed:
        # Generate an additional plot showing correct/incorrect predictions
        fig, ax = plt.subplots(figsize=(15, 8))
        
        correct_mask = pred_classes == true_classes
        incorrect_mask = ~correct_mask
        
        # Plot prediction correctness
        ax.scatter(np.array(epochs)[correct_mask], np.ones(sum(correct_mask)), 
                  color='green', label='Correct', s=50, alpha=0.6)
        ax.scatter(np.array(epochs)[incorrect_mask], np.zeros(sum(incorrect_mask)), 
                  color='red', label='Incorrect', s=50, alpha=0.6)
        
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Incorrect', 'Correct'], fontsize=14)
        ax.set_xlabel('Epoch Number', fontsize=14)
        ax.set_title(f'Prediction Accuracy by Epoch - {os.path.splitext(filename)[0]}', fontsize=16)
        ax.grid(True, axis='y', linestyle='--')
        ax.legend(fontsize=12)
        
        # Save accuracy plot
        acc_output_path = os.path.join(HYPNOGRAM_DIR, f"{os.path.splitext(filename)[0]}_accuracy.png")
        plt.savefig(acc_output_path, dpi=150)
        plt.close(fig)
        logging.info(f"Accuracy plot saved to {acc_output_path}")
    
    return accuracy

def generate_confusion_matrix(true_classes, pred_classes, filename):
    """Generate and save confusion matrix."""
    cm = confusion_matrix(true_classes, pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=SLEEP_STAGES,
               yticklabels=SLEEP_STAGES)
    plt.title(f'Confusion Matrix - {filename}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    output_path = os.path.join(HYPNOGRAM_DIR, f"{os.path.splitext(filename)[0]}_confusion.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logging.info(f"Confusion matrix saved to {output_path}")

def process_file_standard(model, filename):
    """Process a file with standard model (non-temporal)."""
    try:
        # Load data
        mel_path = os.path.join(MEL_SPECTROGRAM_DIR, filename)
        label_path = os.path.join(LABEL_DIR, filename)
        
        logging.info(f"Processing file with standard model: {filename}")
        
        mel_data = np.load(mel_path)
        true_labels = np.load(label_path)
        
        logging.info(f"File contains {len(mel_data)} epochs")
        
        # Process each epoch separately for clarity
        all_predictions = []
        
        # Process in small batches to avoid memory issues
        batch_size = 16  # Smaller batch size for memory efficiency
        for i in range(0, len(mel_data), batch_size):
            batch = mel_data[i:min(i+batch_size, len(mel_data))]
            
            # Preprocess each spectrogram in the batch
            processed_batch = np.array([preprocess_spectrogram(spec) for spec in batch])
            
            # Get predictions for the batch
            batch_predictions = model.predict(processed_batch, verbose=0)
            all_predictions.append(batch_predictions)
            
            # Log progress for large files
            if len(mel_data) > 100 and (i + batch_size) % 100 == 0:
                logging.info(f"  Processed {min(i + batch_size, len(mel_data))}/{len(mel_data)} epochs")
        
        # Combine all predictions
        predictions = np.vstack(all_predictions)
        
        return predictions, true_labels
    
    except Exception as e:
        logging.error(f"Error processing file {filename} with standard model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def process_file_temporal(model, filename):
    """Process a file with temporal model."""
    # Reduce batch size significantly 
    batch_size = 4  # Changed from 16 to 4
    
    # Other optimizations in the function:
    try:
        # Load data with memory mapping
        mel_path = os.path.join(MEL_SPECTROGRAM_DIR, filename)
        label_path = os.path.join(LABEL_DIR, filename)
        
        logging.info(f"Processing file with temporal model: {filename}")
        
        # Use memory mapping for both files
        mel_data = np.load(mel_path, mmap_mode='r')  # Ensure mmap_mode='r' is used
        true_labels = np.load(label_path, mmap_mode='r')
        
        logging.info(f"File contains {len(mel_data)} epochs")
        
        # Skip if file is too short for temporal context
        if len(mel_data) < CONTEXT_SIZE:
            logging.warning(f"File {filename} has fewer segments ({len(mel_data)}) "
                           f"than context size ({CONTEXT_SIZE}). Skipping.")
            return None, None
        
        # Calculate valid predictions (need context on both sides)
        half_context = CONTEXT_SIZE // 2
        valid_indices = range(half_context, len(mel_data) - half_context)
        valid_labels = true_labels[half_context:len(mel_data) - half_context]
        
        all_predictions = []
        
        # Process in smaller batches with memory cleanup
        for i in range(0, len(valid_indices), batch_size):
            batch_indices = valid_indices[i:min(i+batch_size, len(valid_indices))]
            batch_sequences = []
            
            for center_idx in batch_indices:
                # Extract sequence centered at current position
                sequence = []
                for j in range(center_idx - half_context, center_idx + half_context + 1):
                    spec = mel_data[j].copy()
                    # Preprocess
                    if spec.ndim == 2:
                        spec = np.expand_dims(spec, axis=-1)
                        if spec.shape[-1] == 1:
                            spec = np.repeat(spec, 3, axis=-1)
                    
                    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
                    resized_spec = tf.image.resize(spec, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
                    sequence.append(resized_spec.numpy())
                
                batch_sequences.append(np.array(sequence))
            
            # Make predictions on the batch
            batch_data = np.array(batch_sequences)
            batch_predictions = model.predict(batch_data, verbose=0)
            all_predictions.append(batch_predictions)
            
            # After processing each batch, force garbage collection
            gc.collect()
            if tf.config.list_physical_devices('GPU'):
                tf.keras.backend.clear_session()
            
            # Log progress
            if len(valid_indices) > 100 and (i + batch_size) % 100 == 0:
                logging.info(f"  Processed {min(i + batch_size, len(valid_indices))}/{len(valid_indices)} sequences")
        
        # Combine predictions
        predictions = np.vstack(all_predictions) if all_predictions else np.array([])
        
        # Pad predictions with zeros to match original length (first and last half_context epochs)
        # This ensures alignment with true_labels for visualization
        if len(predictions) > 0:
            pad_start = np.zeros((half_context, NUM_CLASSES))
            pad_end = np.zeros((len(mel_data) - len(valid_indices) - half_context, NUM_CLASSES))
            padded_predictions = np.vstack([pad_start, predictions, pad_end])
            return padded_predictions, true_labels
        else:
            return None, None
    
    except Exception as e:
        logging.error(f"Error processing file {filename} with temporal model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def process_file(model, filename):
    """Process each file based on model type."""
    if USE_TEMPORAL:
        predictions, true_labels = process_file_temporal(model, filename)
    else:
        predictions, true_labels = process_file_standard(model, filename)
    
    if predictions is None or true_labels is None:
        return {
            'filename': filename,
            'error': "Failed to process file"
        }
    
    # Generate visualization showing predictions for each epoch
    accuracy = generate_hypnogram(predictions, true_labels, filename)
    
    # Generate confusion matrix
    true_classes = np.argmax(true_labels, axis=1)
    pred_classes = np.argmax(predictions, axis=1)
    generate_confusion_matrix(true_classes, pred_classes, filename)
    
    # Calculate per-class metrics
    class_accuracies = {}
    class_counts = {}
    for i in range(NUM_CLASSES):
        mask = true_classes == i
        count = np.sum(mask)
        if count > 0:
            class_acc = np.mean(pred_classes[mask] == i)
            class_accuracies[SLEEP_STAGES[i]] = class_acc
            class_counts[SLEEP_STAGES[i]] = count
    
    return {
        'filename': filename,
        'accuracy': accuracy,
        'num_epochs': len(true_labels),
        'class_distribution': class_counts,
        'class_accuracies': class_accuracies
    }

def main():
    """Main function to evaluate model on training data."""
    parser = argparse.ArgumentParser(description="Test sleep stage prediction model on training data.")
    parser.add_argument('--file', type=str, help="Process a specific file only")
    args = parser.parse_args()
    
    try:
        # Configure GPU
        configure_gpu()
        
        # Load model with the exact same architecture and saved weights
        model = load_model()
        logging.info(f"Model loaded successfully - Running evaluation on training data")
        
        # Get training files to process
        if args.file:
            if not os.path.exists(os.path.join(MEL_SPECTROGRAM_DIR, args.file)) or \
               not os.path.exists(os.path.join(LABEL_DIR, args.file)):
                logging.error(f"Specified file '{args.file}' not found.")
                return
            files = [args.file]
        else:
            files = get_training_files()
        
        logging.info(f"Processing {len(files)} training file(s)")
        
        # Process each file
        results = []
        for filename in files:
            result = process_file(model, filename)
            results.append(result)
            
            if 'accuracy' in result:
                logging.info(f"File: {filename}, Accuracy: {result['accuracy']:.4f}")
                for stage, acc in result.get('class_accuracies', {}).items():
                    logging.info(f"  - {stage}: {acc:.4f} ({result['class_distribution'].get(stage, 0)} epochs)")
        
        # Generate summary report
        if results and any('accuracy' in r for r in results):
            with open(os.path.join(HYPNOGRAM_DIR, 'training_accuracy_summary.txt'), 'w') as f:
                f.write("Sleep Stage Prediction Results on Training Data\n")
                f.write("=========================================\n\n")
                
                # Calculate overall accuracy
                total_epochs = sum(r.get('num_epochs', 0) for r in results if 'accuracy' in r)
                weighted_acc = sum(r.get('accuracy', 0) * r.get('num_epochs', 0) 
                                  for r in results if 'accuracy' in r)
                
                if total_epochs > 0:
                    overall_acc = weighted_acc / total_epochs
                    f.write(f"Overall accuracy on training data: {overall_acc:.4f} across {total_epochs} epochs\n\n")
                
                # File details
                for r in results:
                    if 'accuracy' in r:
                        f.write(f"File: {r['filename']}\n")
                        f.write(f"  Accuracy: {r['accuracy']:.4f}\n")
                        f.write(f"  Total epochs: {r['num_epochs']}\n")
                        f.write("  Class distribution:\n")
                        for stage in SLEEP_STAGES:
                            count = r['class_distribution'].get(stage, 0)
                            pct = 100 * count / r['num_epochs'] if r['num_epochs'] > 0 else 0
                            f.write(f"    - {stage}: {count} ({pct:.1f}%)\n")
                        
                        f.write("  Class accuracies:\n")
                        for stage in SLEEP_STAGES:
                            if stage in r.get('class_accuracies', {}):
                                f.write(f"    - {stage}: {r['class_accuracies'][stage]:.4f}\n")
                        f.write("\n")
                    else:
                        f.write(f"File: {r['filename']} - Error: {r.get('error', 'Unknown error')}\n\n")
            
            logging.info(f"Summary saved to {os.path.join(HYPNOGRAM_DIR, 'training_accuracy_summary.txt')}")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_tensorflow_resources()

if __name__ == "__main__":
    main()