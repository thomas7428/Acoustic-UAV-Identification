"""
Project Configuration File
Centralized path configuration for all scripts in the project.
Works on both Linux and Windows.
"""

import os
from pathlib import Path

# Project root directory (where this config.py file is located)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Dataset paths
# Use DATASET_ROOT_OVERRIDE environment variable to switch between datasets
# Options: "dataset_test" (original), "dataset_augmented" (augmented only), "dataset_combined" (both)
DEFAULT_DATASET = "dataset_combined"  # Use combined dataset by default
DATASET_NAME = os.environ.get("DATASET_ROOT_OVERRIDE", DEFAULT_DATASET)
DATASET_ROOT = PROJECT_ROOT / "0 - DADS dataset extraction" / DATASET_NAME

# Config of datasets used for training and validation
CONFIG_DATASET_PATH = PROJECT_ROOT / "0 - DADS dataset extraction" / "augment_config_v3.json"

# Feature extraction paths
EXTRACTED_FEATURES_DIR = PROJECT_ROOT / "0 - DADS dataset extraction" / "extracted_features"

# Model save directory
MODELS_DIR = PROJECT_ROOT / "0 - DADS dataset extraction" / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

# Results save directory
RESULTS_DIR = PROJECT_ROOT / "0 - DADS dataset extraction" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# CNN Model paths
CNN_MODEL_PATH = MODELS_DIR / "cnn_model.keras"
CNN_HISTORY_PATH = RESULTS_DIR / "cnn_history.csv"
CNN_ACC_PATH = RESULTS_DIR / "cnn_accuracy.json"

# RNN Model paths
RNN_MODEL_PATH = MODELS_DIR / "rnn_model.keras"
RNN_HISTORY_PATH = RESULTS_DIR / "rnn_history.csv"
RNN_ACC_PATH = RESULTS_DIR / "rnn_accuracy.json"

# CRNN Model paths
CRNN_MODEL_PATH = MODELS_DIR / "crnn_model.keras"
CRNN_HISTORY_PATH = RESULTS_DIR / "crnn_history.csv"
CRNN_ACC_PATH = RESULTS_DIR / "crnn_accuracy.json"

# Attention-Enhanced CRNN Model paths
ATTENTION_CRNN_MODEL_PATH = MODELS_DIR / "attention_crnn_model.keras"
ATTENTION_CRNN_HISTORY_PATH = RESULTS_DIR / "attention_crnn_history.csv"
ATTENTION_CRNN_ACC_PATH = RESULTS_DIR / "attention_crnn_accuracy.json"

# All models accuracy path
ALL_MODELS_ACC_PATH = RESULTS_DIR / "all_model_acc.json"

# Ensemble/Voting model paths (10 models for late fusion)
ENSEMBLE_MODELS_DIR = MODELS_DIR / "ensemble"
ENSEMBLE_MODELS_DIR.mkdir(exist_ok=True)

ENSEMBLE_MODEL_PATHS = {
    1: ENSEMBLE_MODELS_DIR / "model_1.keras",
    2: ENSEMBLE_MODELS_DIR / "model_2.keras",
    3: ENSEMBLE_MODELS_DIR / "model_3.keras",
    4: ENSEMBLE_MODELS_DIR / "model_4.keras",
    5: ENSEMBLE_MODELS_DIR / "model_5.keras",
    6: ENSEMBLE_MODELS_DIR / "model_6.keras",
    7: ENSEMBLE_MODELS_DIR / "model_7.keras",
    8: ENSEMBLE_MODELS_DIR / "model_8.keras",
    9: ENSEMBLE_MODELS_DIR / "model_9.keras",
    10: ENSEMBLE_MODELS_DIR / "model_10.keras",
}

# Performance calculation paths
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)

CNN_PREDICTIONS_PATH = PREDICTIONS_DIR / "cnn_predictions.json"
RNN_PREDICTIONS_PATH = PREDICTIONS_DIR / "rnn_predictions.json"
CRNN_PREDICTIONS_PATH = PREDICTIONS_DIR / "crnn_predictions.json"
ATTENTION_CRNN_PREDICTIONS_PATH = PREDICTIONS_DIR / "attention_crnn_predictions.json"

CNN_SCORES_PATH = PREDICTIONS_DIR / "cnn_scores.json"
RNN_SCORES_PATH = PREDICTIONS_DIR / "rnn_scores.json"
CRNN_SCORES_PATH = PREDICTIONS_DIR / "crnn_scores.json"
ATTENTION_CRNN_SCORES_PATH = PREDICTIONS_DIR / "attention_crnn_scores.json"

# Voting results paths
VOTING_DIR = RESULTS_DIR / "voting"
VOTING_DIR.mkdir(exist_ok=True)

VOTED_RESULTS_PATH = VOTING_DIR / "voted_results.json"
VOTING_WEIGHTS_PATH = VOTING_DIR / "weights.json"
VOTING_FINAL_SCORES_PATH = VOTING_DIR / "FINAL_SCORES.json"

# Individual voted results paths (for ensemble models)
VOTED_PATHS = {
    1: VOTING_DIR / "voted_1.json",
    2: VOTING_DIR / "voted_2.json",
    3: VOTING_DIR / "voted_3.json",
    4: VOTING_DIR / "voted_4.json",
    5: VOTING_DIR / "voted_5.json",
    6: VOTING_DIR / "voted_6.json",
    7: VOTING_DIR / "voted_7.json",
    8: VOTING_DIR / "voted_8.json",
    9: VOTING_DIR / "voted_9.json",
    10: VOTING_DIR / "voted_10.json",
}

SCORES_PATHS = {
    1: VOTING_DIR / "scores_1.json",
    2: VOTING_DIR / "scores_2.json",
    3: VOTING_DIR / "scores_3.json",
    4: VOTING_DIR / "scores_4.json",
    5: VOTING_DIR / "scores_5.json",
    6: VOTING_DIR / "scores_6.json",
    7: VOTING_DIR / "scores_7.json",
    8: VOTING_DIR / "scores_8.json",
    9: VOTING_DIR / "scores_9.json",
    10: VOTING_DIR / "scores_10.json",
}

# Audio parameters
SAMPLE_RATE = 22050
"""
Desired audio duration (seconds) used across the pipeline. Set this to
the target temporal length for all generated/processed WAV files.
Example: 4.0 -> all files should be normalized to 4 seconds.
"""

# Add a single source-of-truth duration constant and keep backwards
# compatible integer `DURATION` used by some scripts.
AUDIO_DURATION_S = 4.0

# Backwards-compatible `DURATION` (integer seconds)
DURATION = int(AUDIO_DURATION_S)
NUM_SEGMENTS = 10

# WAV write subtype used by augmentation/save routines (must be a valid subtype for soundfile)
# Examples: 'PCM_16', 'FLOAT'
AUDIO_WAV_SUBTYPE = 'FLOAT'

# Mel extraction parameters (central source of truth)
# Use these values everywhere to ensure consistent features between
# training, extraction and inference.
MFCC_N_MFCC = 20            # number of MFCCs to extract
MEL_DURATION = float(AUDIO_DURATION_S)        # seconds used per example at training/inference
MEL_N_MELS = 44           # number of mel bins
MEL_N_FFT = 2048          # FFT window size
MEL_HOP_LENGTH = 512      # hop length in samples
MEL_TIME_FRAMES = 90      # number of time frames to pad/truncate to
MEL_PAD_VALUE = -100.0    # sentinel padding value used in precomputed features

# Feature extraction behaviors
# When True, training extraction will enable SpecAugment by default unless
# overridden on the command line in the extractor scripts.
SPEC_AUGMENT_BY_DEFAULT_FOR_TRAIN = True

# Output filenames for per-split extracted features. Use .format(split=...) to
# produce concrete filenames. The extractors will write e.g. mel_train.json.
MEL_OUTPUT_PATTERN = EXTRACTED_FEATURES_DIR / "mel_{split}.json"
MFCC_OUTPUT_PATTERN = EXTRACTED_FEATURES_DIR / "mfcc_{split}.json"

# Precomputed MEL index paths (one MEL per WAV). Evaluation code expects at
# least `mel_test_index.json` when PRECOMPUTED_ONLY=True.
MEL_TEST_DATA_PATH = EXTRACTED_FEATURES_DIR / "mel_test.json"
MEL_TEST_INDEX_PATH = EXTRACTED_FEATURES_DIR / "mel_test_index.json"
MEL_TRAIN_DATA_PATH = EXTRACTED_FEATURES_DIR / "mel_train.json"
MEL_TRAIN_INDEX_PATH = EXTRACTED_FEATURES_DIR / "mel_train_index.json"
MEL_VAL_DATA_PATH = EXTRACTED_FEATURES_DIR / "mel_val.json"
MEL_VAL_INDEX_PATH = EXTRACTED_FEATURES_DIR / "mel_val_index.json"

# Model inference configuration
# Adaptive thresholds for each model (auto-calibrated or manually set)
MODEL_THRESHOLDS = {
    "CNN": 0.5,      # Default threshold
    "RNN": 0.5,      # Will be auto-calibrated if AUTO_CALIBRATE_THRESHOLDS=True
    "CRNN": 0.5,
    "Attention_CRNN": 0.5,
}

# Auto-calibration settings
AUTO_CALIBRATE_THRESHOLDS = False  # Set to True to auto-calibrate on first performance calculation
TARGET_RECALL = 0.95  # Target recall for threshold calibration (prefer false positives over false negatives)

# Enforce usage of precomputed features only. When True, any code path that would
# compute MELs on-the-fly should instead read `mel_test_index.json` or fail.
PRECOMPUTED_ONLY = True

# Ensemble voting configuration
# PHASE 2A: Optimized weights based on distance performance analysis
# CRNN performs best at extreme distances (68% @ 100m vs CNN 15%)
ENSEMBLE_WEIGHTS = {
    "CNN": 0.30,   # Good at close range
    "RNN": 0.10,   # Weakest performer (0% across all distances in Phase 1)
    "CRNN": 0.60,  # Best at long distances, prioritize this model
}

# Distance-adaptive weights (optional, requires distance estimation)
DISTANCE_ADAPTIVE_WEIGHTS = False
DISTANCE_WEIGHT_CONFIG = {
    "extreme": {"CNN": 0.2, "RNN": 0.0, "CRNN": 0.8},  # > 300m
    "long": {"CNN": 0.3, "RNN": 0.1, "CRNN": 0.6},     # 150-300m
    "medium": {"CNN": 0.4, "RNN": 0.2, "CRNN": 0.4},   # 75-150m
    "close": {"CNN": 0.4, "RNN": 0.3, "CRNN": 0.3},    # < 75m
}

# Deployment constraints (Raspberry Pi 3B)
RASPBERRY_PI_MODE = False  # Set to True for optimized inference on RPi3
MAX_LATENCY_MS = 3000      # Maximum acceptable latency in milliseconds
ENABLE_GPU = False         # RPi3 doesn't have GPU acceleration
QUANTIZE_MODELS = False    # Enable INT8 quantization for faster inference

# Convert Path objects to strings for compatibility
def get_path_str(path):
    """Convert Path to string with forward slashes for cross-platform compatibility."""
    return str(path).replace(os.sep, "/")


# String versions of paths (for scripts that need strings)
DATASET_ROOT_STR = get_path_str(DATASET_ROOT)
CNN_MODEL_PATH_STR = get_path_str(CNN_MODEL_PATH)
CNN_HISTORY_PATH_STR = get_path_str(CNN_HISTORY_PATH)
CNN_ACC_PATH_STR = get_path_str(CNN_ACC_PATH)
RNN_MODEL_PATH_STR = get_path_str(RNN_MODEL_PATH)
RNN_HISTORY_PATH_STR = get_path_str(RNN_HISTORY_PATH)
RNN_ACC_PATH_STR = get_path_str(RNN_ACC_PATH)
CRNN_MODEL_PATH_STR = get_path_str(CRNN_MODEL_PATH)
CRNN_HISTORY_PATH_STR = get_path_str(CRNN_HISTORY_PATH)
CRNN_ACC_PATH_STR = get_path_str(CRNN_ACC_PATH)
ALL_MODELS_ACC_PATH_STR = get_path_str(ALL_MODELS_ACC_PATH)
CNN_PREDICTIONS_PATH_STR = get_path_str(CNN_PREDICTIONS_PATH)
RNN_PREDICTIONS_PATH_STR = get_path_str(RNN_PREDICTIONS_PATH)
CRNN_PREDICTIONS_PATH_STR = get_path_str(CRNN_PREDICTIONS_PATH)
CNN_SCORES_PATH_STR = get_path_str(CNN_SCORES_PATH)
RNN_SCORES_PATH_STR = get_path_str(RNN_SCORES_PATH)
CRNN_SCORES_PATH_STR = get_path_str(CRNN_SCORES_PATH)
VOTED_RESULTS_PATH_STR = get_path_str(VOTED_RESULTS_PATH)
VOTING_WEIGHTS_PATH_STR = get_path_str(VOTING_WEIGHTS_PATH)
VOTING_FINAL_SCORES_PATH_STR = get_path_str(VOTING_FINAL_SCORES_PATH)

# String versions of feature extraction outputs and index paths
MEL_OUTPUT_PATTERN_STR = get_path_str(MEL_OUTPUT_PATTERN)
MFCC_OUTPUT_PATTERN_STR = get_path_str(MFCC_OUTPUT_PATTERN)
MEL_TEST_DATA_PATH_STR = get_path_str(MEL_TEST_DATA_PATH)
MEL_TEST_INDEX_PATH_STR = get_path_str(MEL_TEST_INDEX_PATH)
MEL_TRAIN_DATA_PATH_STR = get_path_str(MEL_TRAIN_DATA_PATH)
MEL_TRAIN_INDEX_PATH_STR = get_path_str(MEL_TRAIN_INDEX_PATH)
MEL_VAL_DATA_PATH_STR = get_path_str(MEL_VAL_DATA_PATH)
MEL_VAL_INDEX_PATH_STR = get_path_str(MEL_VAL_INDEX_PATH)

# String versions of voted paths dictionaries
VOTED_PATHS_STR = {k: get_path_str(v) for k, v in VOTED_PATHS.items()}
SCORES_PATHS_STR = {k: get_path_str(v) for k, v in SCORES_PATHS.items()}
