"""
Attention-Enhanced CRNN Trainer
================================

Architecture optimisée pour la détection à longue distance (500m-350m) :
- CNN feature extractor
- Temporal Attention (focus sur time-steps avec signal drone)
- Bidirectional LSTM (contexte temporel)
- Channel Attention (focus sur fréquences drone 500-2000Hz)
- Dense classifier

Améliorations vs CRNN standard :
1. Attention temporelle amplifie segments avec signal faible
2. Attention de canal priorise fréquences drone
3. Deeper feature extraction (4 conv layers vs 3)
4. Residual connections pour gradient flow
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from termcolor import colored

# Project config
sys.path.append(str(Path(__file__).parent.parent))
import config

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Use centralized configuration
DATASET_DIR = Path(config.PROJECT_ROOT) / "0 - DADS dataset extraction"
TRAIN_DATA_PATH = Path(config.MEL_TRAIN_DATA_PATH)
VAL_DATA_PATH = Path(config.MEL_VAL_DATA_PATH)
MODEL_SAVE = Path(config.ATTENTION_CRNN_MODEL_PATH)
HISTORY_SAVE = Path(config.ATTENTION_CRNN_HISTORY_PATH)

# Training parameters (respect config when present)
BATCH_SIZE = getattr(config, 'BATCH_SIZE', 16)
LEARNING_RATE = getattr(config, 'LEARNING_RATE', 0.0001)

SAMPLE_RATE = config.SAMPLE_RATE
N_MELS = config.MEL_N_MELS
N_FFT = config.MEL_N_FFT
HOP_LENGTH = config.MEL_HOP_LENGTH
TIME_FRAMES = config.MEL_TIME_FRAMES
PAD_VALUE = config.MEL_PAD_VALUE

def load_data():
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    import librosa
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    # Get the train and val directories
    print(f"[INFO] Loading precomputed features from {VAL_DATA_PATH}")
    with open(VAL_DATA_PATH, 'r') as f:
        val_data = json.load(f)
    val_mels = np.array(val_data.get('mel', []))
    val_labels = np.array(val_data.get('labels', []))
    if len(val_mels) == 0:
        raise RuntimeError(f"Precomputed features file {VAL_DATA_PATH} contains no 'mel' entries")

    print(f"[INFO] Loading precomputed features from {TRAIN_DATA_PATH}")
    with open(TRAIN_DATA_PATH, 'r') as f:
        train_data = json.load(f)
    train_mels = np.array(train_data.get('mel', []))
    train_labels = np.array(train_data.get('labels', []))
    if len(train_mels) == 0:
        raise RuntimeError(f"Precomputed features file {TRAIN_DATA_PATH} contains no 'mel' entries")
    return train_mels, train_labels, val_mels, val_labels
    
def prepare_datasets():
    """Load precomputed mel features and labels, return arrays:
    X_train, y_train, X_val, y_val
    """
    X_train, y_train, X_val, y_val = load_data()

    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    # Filter out samples with missing labels (None)
    train_mask = np.array([lbl is not None for lbl in y_train])
    val_mask = np.array([lbl is not None for lbl in y_val])
    if not np.all(train_mask):
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        print(f"[INFO] Filtered out {np.sum(~train_mask)} training samples with missing labels")
    if not np.all(val_mask):
        X_val = X_val[val_mask]
        y_val = y_val[val_mask]
        print(f"[INFO] Filtered out {np.sum(~val_mask)} validation samples with missing labels")

    return X_train, y_train, X_val, y_val


def build_attention_crnn(input_shape=(44, 90), num_classes=2):
    """
    Build Attention-Enhanced CRNN model.
    
    Architecture:
    1. Deeper CNN (4 layers) with residual connections
    2. Temporal Attention layer
    3. Bidirectional LSTM
    4. Channel Attention layer
    5. Dense classifier
    """
    inputs = layers.Input(shape=input_shape + (1,), name='input')
    
    # ============================================================
    # PART 1: Deeper CNN Feature Extractor with Residual
    # ============================================================
    
    # Block 1
    x = layers.Conv2D(32, (5, 5), padding='same', activation='relu', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(0.2, name='drop1')(x)
    
    # Block 2
    x = layers.Conv2D(64, (5, 5), padding='same', activation='relu', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(0.3, name='drop2')(x)
    
    # Block 3 with residual
    residual = x
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    
    # Adjust residual dimensions for addition
    residual = layers.MaxPooling2D((2, 2))(residual)
    residual = layers.Conv2D(128, (1, 1), padding='same')(residual)
    x = layers.Add(name='residual_add1')([x, residual])
    x = layers.Dropout(0.3, name='drop3')(x)
    
    # Block 4
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.MaxPooling2D((2, 2), name='pool4')(x)
    x = layers.Dropout(0.4, name='drop4')(x)
    
    # ============================================================
    # PART 2: Reshape for RNN (time-distributed)
    # ============================================================
    
    # Shape after CNN: (batch, freq, time, channels)
    # Reshape to: (batch, time, freq*channels) for LSTM
    shape = x.shape
    x = layers.Reshape((shape[2], shape[1] * shape[3]), name='reshape_for_lstm')(x)
    
    # ============================================================
    # PART 3: Temporal Attention Mechanism
    # ============================================================
    
    # Learn which time-steps contain drone signal
    attention_weights = layers.Dense(1, activation='tanh', name='attention_score')(x)
    attention_weights = layers.Flatten(name='attention_flatten')(attention_weights)
    attention_weights = layers.Activation('softmax', name='attention_softmax')(attention_weights)
    attention_weights = layers.RepeatVector(shape[1] * shape[3], name='attention_repeat')(attention_weights)
    attention_weights = layers.Permute([2, 1], name='attention_permute')(attention_weights)
    
    # Apply attention weights
    x = layers.Multiply(name='attention_applied')([x, attention_weights])
    
    # ============================================================
    # PART 4: Bidirectional LSTM
    # ============================================================
    
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3, name='lstm1'),
        name='bi_lstm1'
    )(x)
    
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.3, name='lstm2'),
        name='bi_lstm2'
    )(x)
    
    # ============================================================
    # PART 5: Channel Attention (Frequency Focus)
    # ============================================================
    
    # Global context
    channel_attention = layers.Dense(64, activation='relu', name='channel_dense1')(x)
    channel_attention = layers.Dense(128, activation='sigmoid', name='channel_attention')(channel_attention)
    
    # Apply channel attention
    x = layers.Multiply(name='channel_attention_applied')([x, channel_attention])
    
    # ============================================================
    # PART 6: Classifier
    # ============================================================
    
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5, name='drop_final')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # Build model
    model = models.Model(inputs=inputs, outputs=outputs, name='Attention_CRNN')
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Attention-Enhanced CRNN')
    parser.add_argument('--min_epochs', type=int, default=50, help='Minimum epochs before early stopping')
    parser.add_argument('--stratified-validation', action='store_true', help='Enable distance-stratified validation')
    args = parser.parse_args()
    
    print(colored("\n" + "="*70, "magenta"))
    print(colored("  ATTENTION-ENHANCED CRNN TRAINING", "magenta"))
    print(colored("="*70, "magenta"))
    print(colored(f"[CONFIG] Min epochs: {args.min_epochs}", "yellow"))
    print(colored(f"[CONFIG] Stratified validation: {args.stratified_validation}", "yellow"))
    # Load datasets
    X_train, y_train, X_val, y_val = prepare_datasets()

    # Add channel dimension for Conv2D if necessary
    if X_train.ndim == 3:
        X_train = X_train[..., np.newaxis]
    if X_val.ndim == 3:
        X_val = X_val[..., np.newaxis]

    # Build model using the actual training shape (like CNN_Trainer does)
    if X_train.ndim != 4:
        raise ValueError(f"X_train must be 4D after adding channel dim, got {X_train.shape}")
    n_mels_actual = X_train.shape[1]
    train_tf = X_train.shape[2]

    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train.astype(int), num_classes=2)
    y_val = keras.utils.to_categorical(y_val.astype(int), num_classes=2)

    # Build model
    print(colored("\n[INFO] Building Attention-Enhanced CRNN...", "cyan"))
    model = build_attention_crnn(input_shape=(n_mels_actual, train_tf))
    
    # Compile with recall-focused loss (heavily penalizes missing drones)
    from loss_functions import get_loss_function
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=get_loss_function(loss_type='recall_focused', fn_penalty=50.0),
        metrics=['accuracy']
    )    
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        start_from_epoch=args.min_epochs,
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        MODEL_SAVE,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    
    callbacks_list = [early_stopping, checkpoint]
    
    # Distance-stratified validation
    if args.stratified_validation:
        from distance_stratified_callback import DistanceStratifiedCallback
        stratified_callback = DistanceStratifiedCallback(
            validation_dir=VAL_DATA_PATH,
            model_name="attention_crnn",
            log_dir=str(config.RESULTS_DIR)
        )
        callbacks_list.append(stratified_callback)
        print(colored("[INFO] Distance-stratified validation ENABLED", "green"))
    
    # Train
    print(colored("\n[INFO] Starting training...", "cyan"))
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=1000,
        callbacks=callbacks_list,
        verbose=2
    )
    
    # Save history
    hist_df = pd.DataFrame(history.history)
    os.makedirs(os.path.dirname(HISTORY_SAVE), exist_ok=True)
    hist_df.to_csv(HISTORY_SAVE)
    
    print(colored("\n" + "="*70, "green"))
    print(colored("  ✓ TRAINING COMPLETED", "green"))
    print(colored("="*70, "green"))
    print(colored(f"Model saved: {MODEL_SAVE}", "cyan"))
    print(colored(f"History saved: {HISTORY_SAVE}", "cyan"))
    print(colored(f"Total epochs: {len(history.history['loss'])}", "yellow"))
    print(colored(f"Best val_accuracy: {max(history.history['val_accuracy']):.4f}", "yellow"))


if __name__ == "__main__":
    main()
