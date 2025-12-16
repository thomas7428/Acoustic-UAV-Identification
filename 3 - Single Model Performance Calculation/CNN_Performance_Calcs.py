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
SAVED_MODEL_PATH = config.CNN_MODEL_PATH_STR  # Path of trained model
SAMPLE_RATE = config.SAMPLE_RATE  # Sample rate in Hz.
DURATION = config.DURATION  # Length of audio files fed. Measured in seconds. MUST MATCH TRAINING!
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Precomputed MEL index path (optional)
MEL_TEST_INDEX_PATH = Path(config.EXTRACTED_FEATURES_DIR) / "mel_test_index.json"

# Predictions (1 or 0)
JSON_PATH = config.CNN_PREDICTIONS_PATH_STR  # CNN predictions path
# Performance scores (accuracy, precision, recall, f1 score)
JSON_PERFORMANCE = config.CNN_SCORES_PATH_STR  # CNN scores path

# Prediction of fed audio
class _Class_Predict_Service:
    """Singleton class for keyword spotting inference with trained models.
    :param model: Trained model
    """
    # Model instance cached here.
    model = None
    _instance = None

    # Predict hard values (1 or 0).
    def predict(self, file_path):
        """
        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """

        # Extract mels from testing audio and return probability for drone class.
        log_mel = self.preprocess(file_path)
        # Ensure 4-dim input: (1, n_mels, n_frames, 1)
        X = log_mel[np.newaxis, ..., np.newaxis]

        predictions = self.model.predict(X, verbose=0)
        # If model outputs softmax with 2 units, take prob of index 1 as drone probability.
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            prob_drone = float(predictions[0, 1])
        else:
            prob_drone = float(predictions.flatten()[0])
        return prob_drone

    # Outputs certainty values for soft voting (1-0).
    def preprocess(self, file_path, n_mels=config.MEL_N_MELS, n_fft=config.MEL_N_FFT, hop_length=config.MEL_HOP_LENGTH, num_segments=config.NUM_SEGMENTS):
        """Extract MFCCs from audio file.
        :param file_path (str): Path of audio file
        :param n_mels (int): # of mels to extract
        :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        """

        # For CNN evaluation we expect a single mel extracted for the whole example
        # Load audio and compute mel spectrogram consistent with training config.
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
    """Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # Ensure an instance is created only the first time the factory function is called.
    if _Class_Predict_Service._instance is None:
        _Class_Predict_Service._instance = _Class_Predict_Service()
        # Load model without compiling (skip custom loss function deserialization)
        _Class_Predict_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH, compile=False)
    return _Class_Predict_Service._instance


# Saving results into a json file.
def save_predictions(dataset_path: Path, json_path: str, limit: int = None):
    """
    Generate predictions for files in `dataset_path` and save to `json_path`.
    If `limit` is provided, it limits the number of files evaluated (split evenly across classes).
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
            # Normalize mel_index to mapping name->mel array if necessary
            if isinstance(mel_index, dict) and 'names' in mel_index and 'mels' in mel_index:
                mel_index = {n: np.array(m) for n, m in zip(mel_index['names'], mel_index['mels'])}
        except Exception:
            mel_index = None

    # Prepare model
    kss = Keyword_Spotting_Service()

    results = []
    scores = []

    # Helper to get mel for path
    def get_mel_for_path(path: Path):
        name = path.name
        if mel_index is not None and name in mel_index:
            mel = np.array(mel_index[name])
            # Ensure shape (n_mels, time_frames)
            if mel.ndim == 3:
                mel = mel.squeeze()
            return mel
        else:
            return kss.preprocess(str(path))

    # Batch predict sequentially (small batches to limit memory)
    all_paths = [*files0, *files1]
    batch_size = 32
    for i in range(0, len(all_paths), batch_size):
        batch_paths = all_paths[i:i+batch_size]
        X_batch = []
        for p in batch_paths:
            mel = get_mel_for_path(p)
            X_batch.append(mel)
        X_batch = np.stack(X_batch, axis=0)[..., np.newaxis]
        preds = kss.model.predict(X_batch, verbose=0)
        if preds.ndim > 1 and preds.shape[1] > 1:
            probs = preds[:, 1]
        else:
            probs = preds.flatten()
        for prob in probs:
            scores.append(float(prob))

    # Convert scores to binary predictions using thresholds
    threshold = config.MODEL_THRESHOLDS.get('CNN', 0.5)
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
    # Calculate confusion matrix and metrics using sklearn
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm layout: [[tn, fp],[fn, tp]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = accuracy_score(y_true, y_pred)
    prf = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    # prf: (precision array, recall array, fscore array, support array)
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
    parser = argparse.ArgumentParser(description='Evaluate CNN on dataset_test')
    parser.add_argument('--limit', type=int, default=0, help='Limit total number of files to evaluate (split across classes)')
    args = parser.parse_args()

    limit = args.limit if args.limit and args.limit > 0 else None
    labels, preds_bin, scores, filenames = save_predictions(DATASET_PATH, config.CNN_PREDICTIONS_PATH_STR, limit=limit)
    perf = performance_calcs(config.CNN_SCORES_PATH_STR, labels, preds_bin)

    print(colored(f"CNN model performance scores have been saved to {config.CNN_SCORES_PATH_STR}.", "green"))
    print(json.dumps(perf, indent=2))
