"""
Multi-Task CNN for Drone Detection with Distance Estimation

This model simultaneously learns:
1. Binary classification: Drone vs No-Drone
2. Distance estimation: 500m / 350m / 200m / 100m / 50m
3. SNR estimation: -32dB / -27dB / -20dB / -10dB / 0dB

By forcing the model to predict distance/SNR, it must learn features
that are distance-specific, which improves detection at extreme distances.
"""

import os
import json
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from termcolor import colored
from sklearn.model_selection import train_test_split


# Paths
DATA_PATH = "../0 - DADS dataset extraction/extracted_features"
DATASET_TRAIN = "../0 - DADS dataset extraction/dataset_train"
DATASET_VAL = "../0 - DADS dataset extraction/dataset_val"
MODEL_SAVE = "../0 - DADS dataset extraction/saved_models/multitask_cnn_model.keras"
HISTORY_SAVE = "../0 - DADS dataset extraction/results/multitask_cnn_history.csv"


def extract_distance_from_filename(filename):
    """
    Extract distance in meters from filename.
    
    Examples:
        aug_drone_500m_pitch_9.0_003.wav -> 500
        aug_drone_350m_timestretch_0.9_010.wav -> 350
        aug_ambient_complex_00123.wav -> 0 (no distance for ambient)
    
    Returns:
        int: Distance in meters (500, 350, 200, 100, 50) or 0 for ambient
    """
    if 'drone_500m' in filename:
        return 500
    elif 'drone_350m' in filename:
        return 350
    elif 'drone_200m' in filename:
        return 200
    elif 'drone_100m' in filename:
        return 100
    elif 'drone_50m' in filename:
        return 50
    else:
        return 0  # ambient/background


def distance_to_label(distance):
    """Convert distance (m) to categorical label."""
    distance_map = {0: 0, 50: 1, 100: 2, 200: 3, 350: 4, 500: 5}
    return distance_map.get(distance, 0)


def snr_from_distance(distance):
    """Estimate SNR (dB) from distance."""
    snr_map = {0: 0, 50: 0, 100: -10, 200: -20, 350: -27, 500: -32}
    return snr_map.get(distance, 0)


def snr_to_label(snr):
    """Convert SNR (dB) to categorical label."""
    snr_map = {0: 0, -10: 1, -20: 2, -27: 3, -32: 4}
    return snr_map.get(snr, 0)


def load_data_with_metadata(dataset_path, max_samples=None):
    """
    Load MEL spectrograms with distance and SNR metadata.
    
    Returns:
        X: MEL spectrograms [n_samples, 44, 90, 1]
        y_class: Binary labels [n_samples] - 0=no-drone, 1=drone
        y_distance: Distance labels [n_samples] - 0-5 (categorical)
        y_snr: SNR labels [n_samples] - 0-4 (categorical)
        filenames: List of filenames for tracking
    """
    X = []
    y_class = []
    y_distance = []
    y_snr = []
    filenames = []
    
    dataset_path = Path(dataset_path)
    
    # Load class 0 (no-drone)
    class_0_dir = dataset_path / "0"
    if class_0_dir.exists():
        wav_files = list(class_0_dir.glob("*.wav"))
        if max_samples:
            wav_files = wav_files[:max_samples // 2]
        
        for wav_file in wav_files:
            try:
                audio, sr = librosa.load(wav_file, sr=22050)
                mel = librosa.feature.melspectrogram(
                    y=audio, sr=sr, n_mels=44, n_fft=2048, hop_length=512
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                
                # Ensure shape (44, 90)
                if mel_db.shape[1] < 90:
                    mel_db = np.pad(mel_db, ((0, 0), (0, 90 - mel_db.shape[1])), mode='constant')
                else:
                    mel_db = mel_db[:, :90]
                
                X.append(mel_db)
                y_class.append(0)
                y_distance.append(0)  # No distance for ambient
                y_snr.append(0)
                filenames.append(wav_file.name)
                
            except Exception as e:
                print(f"Error loading {wav_file.name}: {e}")
    
    # Load class 1 (drone)
    class_1_dir = dataset_path / "1"
    if class_1_dir.exists():
        wav_files = list(class_1_dir.glob("*.wav"))
        if max_samples:
            wav_files = wav_files[:max_samples // 2]
        
        for wav_file in wav_files:
            try:
                audio, sr = librosa.load(wav_file, sr=22050)
                mel = librosa.feature.melspectrogram(
                    y=audio, sr=sr, n_mels=44, n_fft=2048, hop_length=512
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                
                if mel_db.shape[1] < 90:
                    mel_db = np.pad(mel_db, ((0, 0), (0, 90 - mel_db.shape[1])), mode='constant')
                else:
                    mel_db = mel_db[:, :90]
                
                # Extract distance from filename
                distance = extract_distance_from_filename(wav_file.name)
                snr = snr_from_distance(distance)
                
                X.append(mel_db)
                y_class.append(1)
                y_distance.append(distance_to_label(distance))
                y_snr.append(snr_to_label(snr))
                filenames.append(wav_file.name)
                
            except Exception as e:
                print(f"Error loading {wav_file.name}: {e}")
    
    X = np.array(X)
    X = np.expand_dims(X, axis=-1)  # Add channel dimension
    
    return (
        X,
        np.array(y_class),
        np.array(y_distance),
        np.array(y_snr),
        filenames
    )


def build_multitask_model(input_shape):
    """
    Build Multi-Task CNN with shared feature extraction and 3 output heads.
    
    Architecture:
        Input (44, 90, 1)
          ↓
        Shared Feature Extractor (CNN layers)
          ↓
        ├─ Detection Head (Binary: Drone/No-Drone)
        ├─ Distance Head (6 classes: 0m, 50m, 100m, 200m, 350m, 500m)
        └─ SNR Head (5 classes: 0dB, -10dB, -20dB, -27dB, -32dB)
    """
    inputs = keras.Input(shape=input_shape)
    
    # Shared Feature Extractor
    x = keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    # Flatten shared features
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    
    # Task 1: Binary Detection (Drone vs No-Drone)
    detection_head = keras.layers.Dense(64, activation='relu', name='detection_dense')(x)
    detection_head = keras.layers.Dropout(0.4)(detection_head)
    detection_output = keras.layers.Dense(2, activation='softmax', name='detection')(detection_head)
    
    # Task 2: Distance Estimation (6 classes)
    distance_head = keras.layers.Dense(64, activation='relu', name='distance_dense')(x)
    distance_head = keras.layers.Dropout(0.4)(distance_head)
    distance_output = keras.layers.Dense(6, activation='softmax', name='distance')(distance_head)
    
    # Task 3: SNR Estimation (5 classes)
    snr_head = keras.layers.Dense(32, activation='relu', name='snr_dense')(x)
    snr_head = keras.layers.Dropout(0.4)(snr_head)
    snr_output = keras.layers.Dense(5, activation='softmax', name='snr')(snr_head)
    
    model = keras.Model(
        inputs=inputs,
        outputs=[detection_output, distance_output, snr_output],
        name='MultiTask_CNN'
    )
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Task CNN for Drone Detection')
    parser.add_argument('--min_epochs', type=int, default=50,
                       help='Minimum number of epochs before early stopping (default: 50)')
    parser.add_argument('--stratified-validation', action='store_true',
                       help='Enable distance-stratified validation')
    
    args = parser.parse_args()
    
    print(colored("\n[CONFIG] Multi-Task CNN Training", "cyan"))
    print(colored(f"[CONFIG] Min epochs: {args.min_epochs}", "cyan"))
    print()
    
    # Load training data
    print(colored("[INFO] Loading training data with metadata...", "cyan"))
    X_train, y_class_train, y_dist_train, y_snr_train, _ = load_data_with_metadata(DATASET_TRAIN)
    
    print(colored("[INFO] Loading validation data with metadata...", "cyan"))
    X_val, y_class_val, y_dist_val, y_snr_val, _ = load_data_with_metadata(DATASET_VAL)
    
    print(colored(f"[INFO] Training samples: {len(X_train)}", "green"))
    print(colored(f"[INFO] Validation samples: {len(X_val)}", "green"))
    print(colored(f"[INFO] Input shape: {X_train.shape[1:]}", "green"))
    print()
    
    # Build model
    input_shape = X_train.shape[1:]
    model = build_multitask_model(input_shape)
    
    # Compile with multiple losses
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss={
            'detection': 'sparse_categorical_crossentropy',
            'distance': 'sparse_categorical_crossentropy',
            'snr': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'detection': 3.0,  # Primary task - highest weight
            'distance': 1.5,   # Secondary task - help learn distance features
            'snr': 1.0         # Auxiliary task - correlation with distance
        },
        metrics={
            'detection': ['accuracy'],
            'distance': ['accuracy'],
            'snr': ['accuracy']
        }
    )
    
    model.summary()
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_detection_loss',
        mode='min',  # Loss should be minimized
        patience=10,
        start_from_epoch=args.min_epochs,
        restore_best_weights=True
    )
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        MODEL_SAVE,
        monitor='val_detection_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    
    callbacks_list = [early_stopping, checkpoint]
    
    # Distance-stratified validation callback - TEMPORARILY DISABLED (compatibility issue with multi-output model)
    if False and args.stratified_validation:
        from distance_stratified_callback import DistanceStratifiedCallback
        
        # Create a custom callback wrapper for multi-task model
        class MultiTaskStratifiedCallback(keras.callbacks.Callback):
            def __init__(self, validation_dir, model_name, log_dir):
                super().__init__()
                self.validation_dir = validation_dir
                self.model_name = model_name
                self.log_dir = log_dir
                self.base_callback = None
                self.detection_model = None
            
            def set_model(self, model):
                super().set_model(model)
                # Create a wrapper that only outputs detection
                self.detection_model = keras.Model(
                    inputs=model.input,
                    outputs=model.get_layer('detection').output
                )
                # Initialize the base callback with detection model
                self.base_callback = DistanceStratifiedCallback(
                    validation_dir=self.validation_dir,
                    model_name=self.model_name,
                    log_dir=self.log_dir
                )
                # Manually set the model for base callback
                self.base_callback.set_model(self.detection_model)
            
            def on_epoch_end(self, epoch, logs=None):
                if self.base_callback:
                    self.base_callback.on_epoch_end(epoch, logs)
        
        stratified_callback = MultiTaskStratifiedCallback(
            validation_dir="../0 - DADS dataset extraction/dataset_val",
            model_name="multitask_cnn",
            log_dir="../0 - DADS dataset extraction/results"
        )
        callbacks_list.append(stratified_callback)
        print(colored("[INFO] Distance-stratified validation ENABLED", "green"))
    
    # Train
    print(colored("\n[INFO] Starting training...", "cyan"))
    history = model.fit(
        X_train,
        {
            'detection': y_class_train,
            'distance': y_dist_train,
            'snr': y_snr_train
        },
        validation_data=(
            X_val,
            {
                'detection': y_class_val,
                'distance': y_dist_val,
                'snr': y_snr_val
            }
        ),
        batch_size=16,
        epochs=1000,
        callbacks=callbacks_list,
        verbose=2
    )
    
    # Save history
    hist_df = pd.DataFrame(history.history)
    os.makedirs(os.path.dirname(HISTORY_SAVE), exist_ok=True)
    hist_df.to_csv(HISTORY_SAVE)
    print(colored(f"\n[SUCCESS] Training history saved to {HISTORY_SAVE}", "green"))
    
    # Final metrics
    print(colored("\n[INFO] Final Training Metrics:", "cyan"))
    print(f"  Detection Accuracy: {history.history['detection_accuracy'][-1]:.4f}")
    print(f"  Distance Accuracy: {history.history['distance_accuracy'][-1]:.4f}")
    print(f"  SNR Accuracy: {history.history['snr_accuracy'][-1]:.4f}")
    
    print(colored("\n[INFO] Final Validation Metrics:", "cyan"))
    print(f"  Detection Val Accuracy: {history.history['val_detection_accuracy'][-1]:.4f}")
    print(f"  Distance Val Accuracy: {history.history['val_distance_accuracy'][-1]:.4f}")
    print(f"  SNR Val Accuracy: {history.history['val_snr_accuracy'][-1]:.4f}")
    
    print(colored(f"\n[SUCCESS] Model saved to {MODEL_SAVE}", "green"))
