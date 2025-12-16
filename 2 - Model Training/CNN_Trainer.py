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
# DATA_PATH = config.MEL_TRAIN_PATH_STR
TRAIN_DATA_PATH = config.MEL_TRAIN_DATA_PATH_STR
TRAIN_INDEX_PATH = config.MEL_TRAIN_INDEX_PATH_STR
VAL_DATA_PATH = config.MEL_VAL_DATA_PATH_STR
VAL_INDEX_PATH = config.MEL_VAL_INDEX_PATH_STR



# Path to save model.
MODEL_SAVE = config.CNN_MODEL_PATH_STR

# Path to save training history and model accuracy performance at end of training.
HISTORY_SAVE = config.CNN_HISTORY_PATH_STR
ACC_SAVE = config.CNN_ACC_PATH_STR

# Training hyperparams (use config when available)
BATCH_SIZE = getattr(config, 'BATCH_SIZE', 16)
LEARNING_RATE = getattr(config, 'LEARNING_RATE', 0.0001)


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
    # Load from stratified directories
    X_train, y_train, X_validation, y_validation = load_data()
    
    # No test split needed
    X_test = X_validation
    y_test = y_validation

    # 3D array.
    X_train = X_train[..., np.newaxis]  # 4-dim array: (# samples, # time steps, # coefficients, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    # Create model
    model = keras.Sequential()

    # 1st convolutional layer.
    model.add(keras.layers.Conv2D(8, (5, 5), activation='relu', input_shape=input_shape))
        # 8 kernals, and 5x5 grid size of kernal
    model.add(keras.layers.MaxPool2D((5, 5), strides=(2, 2), padding='same'))
        # pooling size 5x5
    model.add(keras.layers.BatchNormalization())
        # Batch Normalization allows model to be more accurate and computations are faster.

    # 2nd convolutional layer.
    model.add(keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Flatten the output and feed into dense layer.
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='relu'))
        # 32 = number of neurons
    model.add(keras.layers.Dropout(0.3))
    # Reduces chances of over fitting.

    # Output layer that uses softmax activation.
    model.add(keras.layers.Dense(2, activation='softmax'))
        # 2 neurons --> depends on how many categories we want to predict.

    return model


def predict(model, X, y):
    # Random prediction post-training.
    X = X[np.newaxis, ...]

    prediction = model.predict(X)

    # Extract index with max value.
    predicted_index = np.argmax(prediction, axis=1)
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CNN model with configurable loss function')
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
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets()

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
            validation_dir=VAL_DATA_PATH,
            model_name="cnn",
            log_dir=str(config.RESULTS_DIR)
        )
        callbacks_list.append(stratified_callback)
        print(colored("[INFO] Distance-stratified validation ENABLED", "green"))

    # Build the CNN network.
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # Get loss function based on arguments
    if args.loss == 'focal':
        loss_fn = get_loss_function('focal', alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.loss == 'weighted_bce':
        loss_fn = get_loss_function('weighted_bce', class_weight_drone=args.class_weight)
    elif args.loss == 'recall_focused':
        loss_fn = get_loss_function('recall_focused', fn_penalty=50.0)
        print("[INFO] Using RECALL-FOCUSED LOSS (FN penalty=50x)")
    else:
        loss_fn = 'sparse_categorical_crossentropy'
    
    # Compile the network with improved loss and metrics
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.summary()

    # Train the CNN.
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=BATCH_SIZE, epochs=1000,
                        callbacks=callbacks_list)

    # Save history.
    hist = pd.DataFrame(history.history)

    # Save to csv:
    hist_csv = HISTORY_SAVE
    with open(hist_csv, mode='w') as f:
        hist.to_csv(f)

    print(
        colored("CRNN model has been trained and its training history has been saved to {}.".format(hist_csv), "green"))

    # Evaluate the CNN on the test set.
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # Timer output.
    time = datetime.now() - startTime
    print(time)

    # Make prediction on a random sample.
    X = X_test[100]
    y = y_test[100]
    predict(model, X, y)

    # Save model accuracies on test set (for weight calculations later on).
    accuracy = {
        "model_acc": [],
        "total_train_time": [],
    }

    accuracy["model_acc"].append(test_accuracy)
    accuracy["total_train_time"].append(str(time))

    with open(ACC_SAVE, "w") as fp:
        json.dump(accuracy, fp, indent=4)
