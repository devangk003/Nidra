import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Directory Paths (Modify as per your directory structure)
mel_spectrogram_dir = 'exports/mel_spectrograms'  # Directory for your mel spectrograms
label_dir = 'exports/labels'  # Directory for your labels (npy files)

# Load dataset
def load_data(mel_dir, label_dir):
    mel_files = sorted(os.listdir(mel_dir))  # List all mel spectrograms
    label_files = sorted(os.listdir(label_dir))  # List all labels
    
    mel_data = []
    label_data = []
    
    print("Processing files: ")
    # Load data from files
    for mel_file, label_file in zip(mel_files, label_files):
        print(f"Processing: {mel_file} -> {label_file}")
        
        mel_path = os.path.join(mel_dir, mel_file)
        label_path = os.path.join(label_dir, label_file)
        
        # Load Mel spectrograms (RGB images)
        mel_specs = np.load(mel_path)
        
        # Load labels
        labels = np.load(label_path)
        
        # Adjust the number of frames in the labels to match the Mel spectrograms
        mel_frames = mel_specs.shape[0]  # Number of frames in Mel spectrogram
        label_frames = len(labels)  # Number of frames in RML labels
        
        if mel_frames > label_frames:
            # If Mel spectrogram has more frames, truncate the remaining labels to 'Wake'
            labels = labels.tolist() + [0] * (mel_frames - label_frames)
        elif mel_frames < label_frames:
            # If RML labels have more frames, discard the extra labels
            labels = labels[:mel_frames]
        
        # Append the adjusted data
        mel_data.append(mel_specs)
        label_data.append(labels)
    
    # Convert to numpy arrays
    mel_data = np.array(mel_data)
    label_data = np.array(label_data)
    
    return mel_data, label_data

# Load the data (you can add new data here)
X, y = load_data(mel_spectrogram_dir, label_dir)

# Reshape and normalize data
X = X.astype('float32') / 255.0  # Normalize to [0, 1] range
y = tf.keras.utils.to_categorical(y, num_classes=4)  # One-hot encode labels

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Load pre-trained model for fine-tuning
def load_pretrained_model(input_shape=(64, 128, 3)):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = True  # Unfreeze the base model for fine-tuning
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')  # 4 classes (Wake, REM, NonREM1/2, NonREM3)
    ])
    
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load pre-trained model
model = load_pretrained_model()

# Summary of the model
model.summary()

# Data augmentation using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fine-tuning the model on the new and old data
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=20,
    steps_per_epoch=len(X_train) // 32,
    validation_steps=len(X_val) // 32,
    verbose=2  # Verbose to show progress during training
)

# Save the fine-tuned model
model.save('fine_tuned_sleep_stage_model.h5')

# Plot training history (accuracy & loss)
def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Plot the training history
plot_history(history)
