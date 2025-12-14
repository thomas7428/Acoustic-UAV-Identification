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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configuration
DATASET_DIR = Path("../0 - DADS dataset extraction")
FEATURES_PATH = DATASET_DIR / "extracted_features" / "mel_pitch_shift_9.0.json"
MODEL_SAVE = DATASET_DIR / "saved_models" / "attention_crnn_model.keras"
HISTORY_SAVE = DATASET_DIR / "results" / "attention_crnn_history.csv"

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0001


def load_data():
    """Load data directly from stratified train/val directories."""
    import librosa
    from pathlib import Path
    
    print(colored("[INFO] Loading from dataset_train and dataset_val (stratified by distance)...", "cyan"))
    
    train_dir = Path("../0 - DADS dataset extraction/dataset_train")
    val_dir = Path("../0 - DADS dataset extraction/dataset_val")
    
    def load_from_dir(base_dir, desc):
        X, y = [], []
        
        # Load class 0 (no-drone)
        class_0_dir = base_dir / "0"
        for wav_file in sorted(class_0_dir.glob("*.wav")):
            audio, sr = librosa.load(wav_file, sr=22050, duration=4.0)
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=44, n_fft=2048, hop_length=512)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            if mel_db.shape[1] < 90:
                mel_db = np.pad(mel_db, ((0, 0), (0, 90 - mel_db.shape[1])), mode='constant')
            else:
                mel_db = mel_db[:, :90]
            
            X.append(mel_db)
            y.append(0)
        
        print(f"  [{desc}] Loaded {len(y)} no-drone samples")
        
        # Load class 1 (drone)
        class_1_dir = base_dir / "1"
        drone_count_start = len(y)
        for wav_file in sorted(class_1_dir.glob("*.wav")):
            audio, sr = librosa.load(wav_file, sr=22050, duration=4.0)
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=44, n_fft=2048, hop_length=512)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            if mel_db.shape[1] < 90:
                mel_db = np.pad(mel_db, ((0, 0), (0, 90 - mel_db.shape[1])), mode='constant')
            else:
                mel_db = mel_db[:, :90]
            
            X.append(mel_db)
            y.append(1)
        
        print(f"  [{desc}] Loaded {len(y) - drone_count_start} drone samples")
        
        return np.array(X), np.array(y)
    
    X_train, y_train = load_from_dir(train_dir, "TRAIN")
    X_val, y_val = load_from_dir(val_dir, "VAL")
    
    print(f"\n[INFO] Training: {len(X_train)} samples ({np.sum(y_train==1)} drones, {np.sum(y_train==0)} ambient)")
    print(f"[INFO] Validation: {len(X_val)} samples ({np.sum(y_val==1)} drones, {np.sum(y_val==0)} ambient)")
    print(f"[INFO] Input shape: {X_train.shape[1:]}")
    
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
    
    # Load data (already split by distance in dataset_train/dataset_val)
    X_train, y_train, X_val, y_val = load_data()
    
    # Add channel dimension for Conv2D
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_val = keras.utils.to_categorical(y_val, num_classes=2)
    
    # Build model
    print(colored("\n[INFO] Building Attention-Enhanced CRNN...", "cyan"))
    model = build_attention_crnn(input_shape=(44, 90))
    
    # Compile with recall-focused loss (heavily penalizes missing drones)
    from loss_functions import get_loss_function
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=get_loss_function(loss_type='recall_focused', fn_penalty=50.0),
        metrics=['accuracy']
    )
    
    print(colored("[INFO] Using RECALL-FOCUSED LOSS (FN penalty=50x)", "yellow"))
    
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
            validation_dir="../0 - DADS dataset extraction/dataset_val",
            model_name="attention_crnn",
            log_dir="../0 - DADS dataset extraction/results"
        )
        callbacks_list.append(stratified_callback)
        print(colored("[INFO] Distance-stratified validation ENABLED", "green"))
    
    # Train
    print(colored("\n[INFO] Starting training...", "cyan"))
    print(colored("[INFO] Using stratified validation set (45% @ 500m, 25% @ 350m)", "green"))
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),  # Now uses correct stratified validation
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
