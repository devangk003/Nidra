import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
import logging
import sys
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the SumLayer class again - must match the one used in training
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
        
def build_standard_model(input_shape, num_classes):
    """Build the standard non-temporal model."""
    logging.info("Building standard model...")
    
    try:
        # Match exactly with training model - first try imagenet weights
        try:
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        except tf.errors.ResourceExhaustedError:
            logging.warning("GPU memory exhausted while loading EfficientNetB0.")
            logging.info("Attempting to continue with limited GPU memory...")
            # Try without preloaded weights
            base_model = EfficientNetB0(
                weights=None,
                include_top=False,
                input_shape=input_shape
            )
        
        # Freeze the base model - this is critical!
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    except Exception as e:
        logging.error(f"Error building standard model: {e}")
        sys.exit(1)
        

def build_temporal_model(input_shape, num_classes, context_size):
    """Build a model that processes sequences of spectrograms with temporal context."""
    logging.info(f"Building temporal model with context size {context_size}...")
    
    try:
        # Input shape includes the sequence dimension
        seq_input_shape = (context_size,) + input_shape
        
        # Create input layer
        inputs = tf.keras.Input(shape=seq_input_shape)
        
        # Match exactly with training model - first try imagenet weights
        try:
            cnn = EfficientNetB0(
                weights='imagenet',
                include_top=False, 
                pooling='avg',
                input_shape=input_shape
            )
        except tf.errors.ResourceExhaustedError:
            logging.warning("GPU memory exhausted loading EfficientNetB0 with weights.")
            logging.info("Falling back to model without pre-trained weights...")
            cnn = EfficientNetB0(
                weights=None, 
                include_top=False, 
                pooling='avg',
                input_shape=input_shape
            )
        
        # Freeze CNN weights - this is critical!
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
        attention = layers.Reshape((context_size,))(attention)  
        attention_weights = layers.Activation('softmax')(attention)
        attention_weights = layers.Reshape((context_size, 1))(attention_weights)
        
        # Multiply features by attention weights and sum
        context_vector = layers.Multiply()([x, attention_weights])
        context_vector = SumLayer(axis=1)(context_vector)
        
        # Final classification layers
        x = layers.Dense(128, activation='gelu')(context_vector)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        return model
        
    except Exception as e:
        logging.error(f"Error building temporal model: {e}")
        sys.exit(1)

def load_model(model_dir=None):
    """
    Load model from saved weights and configuration.
    
    Args:
        model_dir: Path to model directory (defaults to auto-detect)
    
    Returns:
        Compiled model ready for inference or training
    """
    # Auto-detect model directory if not specified
    if model_dir is None:
        if os.path.exists('models/temporal/model_config.json'):
            model_dir = 'models/temporal'
        elif os.path.exists('models/standard/model_config.json'):
            model_dir = 'models/standard'
        else:
            raise FileNotFoundError("Could not find model configuration file. Please specify model_dir.")
            
    logging.info(f"Loading model from {model_dir}")
    
    # Load configuration
    try:
        with open(f'{model_dir}/model_config.json', 'r') as f:
            config = json.load(f)
            
        context_size = config.get('context_size', 7)  # Default to 7 if not specified
        input_shape = tuple(config.get('input_shape', (224, 224, 3)))
        num_classes = config.get('num_classes', 4)
        is_temporal = config.get('is_temporal', True)
        
        logging.info(f"Loaded configuration: temporal={is_temporal}, input_shape={input_shape}, "
                    f"num_classes={num_classes}, context_size={context_size}")
    except Exception as e:
        logging.error(f"Error loading model configuration: {e}")
        sys.exit(1)
        
    # Build appropriate model
    if is_temporal:
        model = build_temporal_model(input_shape, num_classes, context_size)
    else:
        model = build_standard_model(input_shape, num_classes)
        
    # Load weights
    try:
        weights_path = f'{model_dir}/weights/final_model_weights.h5'
        model.load_weights(weights_path)
        logging.info(f"Loaded model weights from {weights_path}")
    except Exception as e:
        logging.error(f"Error loading model weights: {e}")
        
        # Try alternative path (checkpoint weights)
        try:
            checkpoint_dir = 'models/temporal_checkpoints' if is_temporal else 'models/checkpoints'
            alt_weights_path = f'{checkpoint_dir}/best_model_weights.h5'
            logging.info(f"Trying alternative weights path: {alt_weights_path}")
            model.load_weights(alt_weights_path)
            logging.info(f"Loaded model weights from {alt_weights_path}")
        except Exception as e:
            logging.error(f"Error loading alternative model weights: {e}")
            sys.exit(1)
            
    # Compile model for inference
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, is_temporal

def verify_model(model):
    """
    Verify the model is working by running a simple prediction
    """
    # Create a small random input for testing
    if model.input_shape[0] is None:  # Batch dimension
        # For temporal model
        sample_shape = model.input_shape[1:]
        dummy_input = np.random.random((1,) + sample_shape)
    else:
        dummy_input = np.random.random((1,) + model.input_shape[1:])
    
    # Try a prediction
    try:
        result = model.predict(dummy_input, verbose=0)
        logging.info(f"Model verification successful! Output shape: {result.shape}")
        return True
    except Exception as e:
        logging.error(f"Model verification failed: {e}")
        return False

# Update main function
if __name__ == "__main__":
    try:
        # Load model and get information about temporal status
        model, is_temporal = load_model()
        
        # Print model summary
        model.summary()
        
        # Verify model works
        if verify_model(model):
            print("\n✅ Model compilation successful!")
            print("The model is loaded and ready for use in memory.")
            print("\nTo use this model for predictions, simply run:")
            print("  predictions = model.predict(your_input_data)")
        else:
            print("\n❌ Model compilation completed but verification failed.")
            print("The model may have issues with predictions.")
        
    except Exception as e:
        logging.error(f"Error compiling model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)