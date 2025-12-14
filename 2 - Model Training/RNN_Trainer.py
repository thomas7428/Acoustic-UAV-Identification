import json
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import pandas as pd
from datetime import datetime
from termcolor import colored
import sys
from pathlib import Path

# Add project root to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Import loss functions
from loss_functions import get_loss_function, get_metrics

# Timer.
startTime = datetime.now()

# Path to created json file from mel preprocess and feature extraction script.
DATA_PATH = config.MEL_TRAIN_PATH_STR

# Path to save model.
MODEL_SAVE = config.RNN_MODEL_PATH_STR

# Path to save training history and model accuracy performance at end of training.
HISTORY_SAVE = config.RNN_HISTORY_PATH_STR
ACC_SAVE = config.RNN_ACC_PATH_STR

def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    import librosa
    from pathlib import Path
    
    print("Loading from stratified directories (45% @ 500m)...")
    train_dir = Path("../0 - DADS dataset extraction/dataset_train")
    val_dir = Path("../0 - DADS dataset extraction/dataset_val")
    
    def load_from_dir(base_dir):
        X, y = [], []
        for label in [0, 1]:
            class_dir = base_dir / str(label)
            for wav_file in sorted(class_dir.glob("*.wav")):
                audio, sr = librosa.load(wav_file, sr=22050, duration=4.0)
                mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=44, n_fft=2048, hop_length=512)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                if mel_db.shape[1] < 90:
                    mel_db = np.pad(mel_db, ((0, 0), (0, 90 - mel_db.shape[1])), mode='constant')
                else:
                    mel_db = mel_db[:, :90]
                X.append(mel_db)
                y.append(label)
        return np.array(X), np.array(y)
    
    X_train, y_train = load_from_dir(train_dir)
    X_val, y_val = load_from_dir(val_dir)
    return X_train, y_train, X_val, y_val

def prepare_datasets(test_size, validation_size):
    X_train, y_train, X_validation, y_validation = load_data(DATA_PATH)
    X_test = X_validation
    y_test = y_validation

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    # Create model.
    model = keras.Sequential()

    # 3 LSTM layers
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True), input_shape=input_shape))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True), input_shape=(input_shape, 1)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(100)))

    # Dense layer
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    # Output layer
    model.add(keras.layers.Dense(2, activation='softmax'))

    return model


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train RNN model with configurable loss function')
    parser.add_argument('--loss', type=str, default='focal', choices=['bce', 'weighted_bce', 'focal'],
                       help='Loss function: bce, weighted_bce, or focal (default: focal)')
    parser.add_argument('--class_weight', type=float, default=3.0,
                       help='Class weight for drone class (for weighted_bce, default: 3.0)')
    parser.add_argument('--focal_alpha', type=float, default=0.75,
                       help='Focal loss alpha parameter (default: 0.75)')
    parser.add_argument('--focal_gamma', type=float, default=0.0,
                       help='Focal loss gamma parameter (default: 0.0) - Standard BCE')
    parser.add_argument('--min_epochs', type=int, default=50,
                       help='Minimum number of epochs before early stopping (default: 50)')
    parser.add_argument('--stratified-validation', action='store_true',
                       help='Enable distance-stratified validation (evaluates each distance separately)')
    
    args = parser.parse_args()
    
    print(colored(f"\n[CONFIG] Loss function: {args.loss}", "cyan"))
    if args.loss == 'weighted_bce':
        print(colored(f"[CONFIG] Drone class weight: {args.class_weight}", "cyan"))
    elif args.loss == 'focal':
        print(colored(f"[CONFIG] Focal alpha: {args.focal_alpha}, gamma: {args.focal_gamma}", "cyan"))
    print()
    
    # Create train, validation and test sets.
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)  # (test size, val size)

    # Early stopping with minimum epochs.
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=args.min_epochs)

    # Checkpoint.
    checkpoint = keras.callbacks.ModelCheckpoint(MODEL_SAVE, monitor='val_loss',
                                                  mode='min', save_best_only=True, verbose=1)
    
    # Distance-stratified validation callback (optional)
    callbacks_list = [callback, checkpoint]
    if args.stratified_validation:
        from distance_stratified_callback import DistanceStratifiedCallback
        stratified_callback = DistanceStratifiedCallback(
            validation_dir="../0 - DADS dataset extraction/dataset_val",
            model_name="rnn",
            log_dir="../0 - DADS dataset extraction/results"
        )
        callbacks_list.append(stratified_callback)
        print(colored("[INFO] Distance-stratified validation ENABLED", "green"))

    # Build the RNN network.
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    # Get loss function based on arguments
    if args.loss == 'focal':
        loss_fn = get_loss_function('focal', alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.loss == 'weighted_bce':
        loss_fn = get_loss_function('weighted_bce', class_weight_drone=args.class_weight)
    else:
        loss_fn = 'sparse_categorical_crossentropy'
    
    # Override with recall_focused for Phase 2F
    loss_fn = get_loss_function('recall_focused', fn_penalty=50.0)
    print("[PHASE 2F] FORCING RECALL-FOCUSED LOSS (FN penalty=50x)")
    
    # Compile the network with improved loss and metrics
    optimiser = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimiser,
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.summary()

    # Train the RNN.
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=64, epochs=1000,
                          callbacks=callbacks_list)

    # Save history.
    hist = pd.DataFrame(history.history)

    # Save to csv:
    hist_csv = HISTORY_SAVE
    with open(hist_csv, mode='w') as f:
        hist.to_csv(f)

    print(colored("Paper 8's model has been trained and its training history has been saved to {}.".format(hist_csv), "green"))

    # Evaluate the RNN on the test set.
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # Timer output.
    time = datetime.now() - startTime
    print(time)

    # Save model accuracies on test set.
    accuracy = {
        "model_acc": [],
        "total_train_time": [],
    }

    accuracy["model_acc"].append(test_accuracy)
    accuracy["total_train_time"].append(str(time))

    with open(ACC_SAVE, "w") as fp:
        json.dump(accuracy, fp, indent=4)
