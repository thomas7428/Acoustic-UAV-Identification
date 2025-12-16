import os
import json
import librosa
import tensorflow as tf
import numpy as np
from termcolor import colored
import sys
from pathlib import Path
import warnings
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Suppress librosa warnings
warnings.filterwarnings('ignore', message='n_fft=.*is too large for input signal of length=.*')

# Import centralized configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Read and save parameters.
# Force dataset_test for performance evaluation (not dataset_combined!)
DATASET_PATH = Path(config.PROJECT_ROOT / "0 - DADS dataset extraction" / "dataset_test")
SAVED_MODEL_PATH = config.RNN_MODEL_PATH_STR  # Path of trained model
SAMPLE_RATE = config.SAMPLE_RATE  # Sample rate in Hz.
DURATION = config.DURATION  # Length of audio files fed. Measured in seconds. MUST MATCH TRAINING!
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Precomputed MEL index path (optional)
MEL_TEST_INDEX_PATH = Path(config.EXTRACTED_FEATURES_DIR) / "mel_test_index.json"

# Predictions (1 or 0)
JSON_PATH = config.RNN_PREDICTIONS_PATH_STR
# Performance scores (accuracy, precision, recall, f1 score)
JSON_PERFORMANCE = config.RNN_SCORES_PATH_STR


# Prediction of fed audio
class _Class_Predict_Service:
    """Singleton class for keyword spotting inference with trained models.
    :param model: Trained model
    """
    model = None
    _instance = None

    # Predict probability for drone class (float between 0 and 1).
    def predict(self, X):
        predictions = self.model.predict(X, verbose=0)
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            probs = predictions[:, 1]
        else:
            probs = predictions.reshape(-1)
        return probs

    # Extract mel for a single file (matches training config)
    def preprocess(self, file_path):
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=config.MEL_DURATION)
        mel = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_mels=config.MEL_N_MELS, n_fft=config.MEL_N_FFT, hop_length=config.MEL_HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Pad or trim to expected time frames
        if mel_db.shape[1] < config.MEL_TIME_FRAMES:
            mel_db = np.pad(mel_db, ((0, 0), (0, config.MEL_TIME_FRAMES - mel_db.shape[1])), mode='constant', constant_values=(config.MEL_PAD_VALUE,))
        else:
            mel_db = mel_db[:, :config.MEL_TIME_FRAMES]
        return mel_db


def Keyword_Spotting_Service():
    """Factory to get singleton model instance"""
    if _Class_Predict_Service._instance is None:
        _Class_Predict_Service._instance = _Class_Predict_Service()
        _Class_Predict_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH, compile=False)
    return _Class_Predict_Service._instance


# Saving results into a json file.
def save_predictions(dataset_path: Path, json_path: str, limit: int = None):
    """Generate predictions for files in `dataset_path` and save to `json_path`.
    Returns: (y_true, preds_bin, scores, filenames)
    """
    # Build file lists explicitly (class 0 then class 1)
    class0_dir = dataset_path / "0"
    class1_dir = dataset_path / "1"
    files0 = sorted([p for p in class0_dir.glob("*.wav")]) if class0_dir.exists() else []
    files1 = sorted([p for p in class1_dir.glob("*.wav")]) if class1_dir.exists() else []

    # Determine per-class limit
    if limit is not None and limit > 0:
        per_class = max(1, limit // 2)
        files0 = files0[:per_class]
        files1 = files1[:per_class]

    filenames = [p.name for p in files0] + [p.name for p in files1]
    labels = [0] * len(files0) + [1] * len(files1)

    # Load precomputed mel index if available and allowed
    mel_index = None
    if config.PRECOMPUTED_ONLY and MEL_TEST_INDEX_PATH.exists():
        try:
            with open(MEL_TEST_INDEX_PATH, 'r') as f:
                mel_index = json.load(f)
            if isinstance(mel_index, dict) and 'names' in mel_index and 'mels' in mel_index:
                mel_index = {n: np.array(m) for n, m in zip(mel_index['names'], mel_index['mels'])}
        except Exception:
            mel_index = None

    kss = Keyword_Spotting_Service()

    results = []
    scores = []

    def get_mel_for_path(path: Path):
        name = path.name
        if mel_index is not None and name in mel_index:
            mel = np.array(mel_index[name])
            if mel.ndim == 3:
                mel = mel.squeeze()
            # Ensure shape (n_mels, time_frames)
            if mel.shape[1] < config.MEL_TIME_FRAMES:
                mel = np.pad(mel, ((0, 0), (0, config.MEL_TIME_FRAMES - mel.shape[1])), mode='constant', constant_values=(config.MEL_PAD_VALUE,))
            else:
                mel = mel[:, :config.MEL_TIME_FRAMES]
            return mel
        else:
            return kss.preprocess(str(path))

    # Batch predict sequentially
    all_paths = [*files0, *files1]
    batch_size = 32
    for i in range(0, len(all_paths), batch_size):
        batch_paths = all_paths[i:i+batch_size]
        X_batch = []
        for p in batch_paths:
            mel = get_mel_for_path(p)
            X_batch.append(mel)
        X_batch = np.stack(X_batch, axis=0)[..., np.newaxis]
        preds = kss.predict(X_batch)
        for prob in preds:
            scores.append(float(prob))

    # Convert scores to binary predictions using thresholds
    threshold = config.MODEL_THRESHOLDS.get('RNN', 0.5)
    preds_bin = [1 if s > threshold else 0 for s in scores]

    data = {
        "mapping": ['0', '1'],
        "names": filenames,
        "results": preds_bin,
        "scores": scores
    }

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=2)

    return labels, preds_bin, scores, filenames


# Calculating performance scores (accuracy, precision, recall, f-score).
def performance_calcs(performance_path, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = accuracy_score(y_true, y_pred)
    prf = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    precision = prf[0].tolist()
    recall = prf[1].tolist()
    fscore = prf[2].tolist()

    performance = {
        "TP": [tp],
        "FN": [fn],
        "TN": [tn],
        "FP": [fp],
        "Accuracy": [acc],
        "Precision": precision,
        "Recall": recall,
        "F1 Score": fscore
    }

    with open(performance_path, "w") as fp:
        json.dump(performance, fp, indent=2)

    return performance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate RNN on dataset_test')
    parser.add_argument('--limit', type=int, default=0, help='Limit total number of files to evaluate (split across classes)')
    args = parser.parse_args()

    limit = args.limit if args.limit and args.limit > 0 else None
    labels, preds_bin, scores, filenames = save_predictions(DATASET_PATH, config.RNN_PREDICTIONS_PATH_STR, limit=limit)
    perf = performance_calcs(config.RNN_SCORES_PATH_STR, labels, preds_bin)

    print(colored(f"RNN model performance scores have been saved to {config.RNN_SCORES_PATH_STR}.", "green"))
    print(json.dumps(perf, indent=2))
