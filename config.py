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

# Feature extraction paths
EXTRACTED_FEATURES_DIR = PROJECT_ROOT / "0 - DADS dataset extraction" / "extracted_features"
MEL_DATA_PATH = EXTRACTED_FEATURES_DIR / "mel_data.json"
MFCC_DATA_PATH = EXTRACTED_FEATURES_DIR / "mfcc_data.json"

# Training data paths (compatibility with original naming)
MEL_TRAIN_PATH = EXTRACTED_FEATURES_DIR / "mel_pitch_shift_9.0.json"
MFCC_TRAIN_PATH = EXTRACTED_FEATURES_DIR / "mfcc_pitch_shift_9.0.json"

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

CNN_SCORES_PATH = PREDICTIONS_DIR / "cnn_scores.json"
RNN_SCORES_PATH = PREDICTIONS_DIR / "rnn_scores.json"
CRNN_SCORES_PATH = PREDICTIONS_DIR / "crnn_scores.json"

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
DURATION = 10
NUM_SEGMENTS = 10

# Convert Path objects to strings for compatibility
def get_path_str(path):
    """Convert Path to string with forward slashes for cross-platform compatibility."""
    return str(path).replace(os.sep, "/")


# String versions of paths (for scripts that need strings)
DATASET_ROOT_STR = get_path_str(DATASET_ROOT)
MEL_DATA_PATH_STR = get_path_str(MEL_DATA_PATH)
MFCC_DATA_PATH_STR = get_path_str(MFCC_DATA_PATH)
MEL_TRAIN_PATH_STR = get_path_str(MEL_TRAIN_PATH)
MFCC_TRAIN_PATH_STR = get_path_str(MFCC_TRAIN_PATH)
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

# String versions of voted paths dictionaries
VOTED_PATHS_STR = {k: get_path_str(v) for k, v in VOTED_PATHS.items()}
SCORES_PATHS_STR = {k: get_path_str(v) for k, v in SCORES_PATHS.items()}
