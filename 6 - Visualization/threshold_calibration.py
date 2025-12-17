"""
Compute optimal decision thresholds per model using precomputed test MELs.

This script loads precomputed MELs from `mel_test_index.json`, loads trained
models, computes per-file positive-class scores for each model, then searches
for thresholds that maximize F1 (and optionally thresholds that meet a target
recall). Results are saved to `6 - Visualization/outputs/model_thresholds.json`
and diagnostic plots are generated.

Usage:
  python threshold_calibration.py

"""
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from tools import plot_utils

# plotting style
plot_utils.set_style()

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from tensorflow import keras
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score


OUTPUT_DIR = plot_utils.get_output_dir(__file__)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_test_index():
    path = config.EXTRACTED_FEATURES_DIR / 'mel_test_index.json'
    if not path.exists():
        raise FileNotFoundError(f"Test index not found: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    names = data.get('names', [])
    mels = data.get('mel', [])
    labels = data.get('labels', None)
    if labels is None or len(labels) != len(names):
        # Try to infer labels from filenames (parent folder info not available here), so abort
        print("[WARNING] labels not present in test index; calibration requires true labels. Aborting.")
        return None
    return {'names': names, 'mels': mels, 'labels': labels}


def load_models():
    model_paths = {
        'CNN': config.CNN_MODEL_PATH,
        'RNN': config.RNN_MODEL_PATH,
        'CRNN': config.CRNN_MODEL_PATH,
        'Attention-CRNN': config.ATTENTION_CRNN_MODEL_PATH
    }
    models = {}
    for name, p in model_paths.items():
        if p.exists():
            try:
                models[name] = keras.models.load_model(p, compile=False)
                print(f"[OK] Loaded {name}")
            except Exception as e:
                print(f"[WARNING] Could not load {name}: {e}")
        else:
            print(f"[SKIP] Model not found: {p}")
    return models


def prepare_inputs_for_model(mel_list, model_name):
    # mel_list: list of (n_mels, time_frames) lists or arrays
    arr = np.array(mel_list).astype(float)
    # shape: (N, n_mels, time)
    if model_name in ['CNN', 'CRNN', 'Attention-CRNN']:
        # add channel dimension
        arr = arr[..., np.newaxis]
    # RNN uses (N, n_mels, time) without channel
    return arr


def compute_scores(model, inputs):
    # Predict in batches to avoid memory issues
    batch_size = 256
    n = len(inputs)
    scores = []
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch = inputs[start:end]
        preds = model.predict(batch, verbose=0)
        # preds could be shape (B,2) or (B,1). Extract positive-class score
        for p in preds:
            p = np.array(p)
            if p.size == 1:
                pos = float(p[0])
            else:
                # assume index 1 is positive class
                pos = float(p[1])
            scores.append(pos)
    return np.array(scores)


def find_best_threshold(y_true, y_scores, target_recall=None):
    # Compute precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # precision_recall_curve returns arrays where thresholds has length len(precisions)-1
    best_f1 = -1.0
    best_t = 0.5
    # Evaluate F1 for thresholds grid
    th_grid = np.linspace(0.0, 1.0, 1001)
    for t in th_grid:
        preds = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    target_t = None
    achieved_recall = None
    if target_recall is not None:
        # Find threshold that yields recall >= target_recall with maximum precision
        cand_t = None
        cand_prec = -1.0
        for t in th_grid:
            preds = (y_scores >= t).astype(int)
            r = recall_score(y_true, preds, zero_division=0)
            p = precision_score(y_true, preds, zero_division=0)
            if r >= target_recall and p > cand_prec:
                cand_prec = p
                cand_t = t
        if cand_t is not None:
            target_t = float(cand_t)
            achieved_recall = float(recall_score(y_true, (y_scores >= target_t).astype(int), zero_division=0))

    return {'best_threshold': best_t, 'best_f1': float(best_f1), 'target_threshold': target_t, 'target_recall': achieved_recall}


def plot_diagnostics(model_name, y_true, y_scores, out_dir):
    plt.figure(figsize=(6, 4))
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    plt.plot(recalls, precisions, label='PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall: {model_name}')
    plt.grid(True)
    plt.tight_layout()
    plot_utils.save_figure(plt.gcf(), f'{model_name}_pr_curve.png', script_path=__file__)
    plt.close()

    # F1 vs threshold
    th_grid = np.linspace(0.0, 1.0, 501)
    f1s = [f1_score(y_true, (y_scores >= t).astype(int), zero_division=0) for t in th_grid]
    plt.figure(figsize=(6, 4))
    plt.plot(th_grid, f1s)
    plt.xlabel('Threshold')
    plt.ylabel('F1')
    plt.title(f'F1 vs Threshold: {model_name}')
    plt.grid(True)
    plt.tight_layout()
    plot_utils.save_figure(plt.gcf(), f'{model_name}_f1_vs_threshold.png', script_path=__file__)
    plt.close()


def main():
    print("[1/4] Loading test index...")
    test_index = load_test_index()
    if test_index is None:
        print("[ERROR] test index loading failed - ensure labels present in mel_test_index.json")
        return 1

    names = test_index['names']
    mels = test_index['mels']
    labels = np.array(test_index['labels'], dtype=int)

    print(f"[OK] Found {len(names)} entries")

    print("[2/4] Loading models...")
    models = load_models()
    if not models:
        print("[ERROR] No models available for calibration")
        return 1

    thresholds_out = {}

    for model_name, model in models.items():
        print(f"[3] Processing model: {model_name}")
        X = prepare_inputs_for_model(mels, model_name)
        print(f"  Input shape: {X.shape}")

        # Compute positive-class scores
        y_scores = compute_scores(model, X)
        y_true = labels

        print("  Computing optimal thresholds...")
        res = find_best_threshold(y_true, y_scores, target_recall=(config.TARGET_RECALL if config.AUTO_CALIBRATE_THRESHOLDS else None))

        thresholds_out[model_name] = res

        # Save diagnostic plots
        plot_diagnostics(model_name, y_true, y_scores, OUTPUT_DIR)

        # Also save raw scores for auditing
        scores_path = OUTPUT_DIR / f"{model_name.lower().replace('-', '_')}_raw_scores.npz"
        np.savez_compressed(scores_path, names=np.array(names), scores=y_scores, labels=y_true)
        print(f"  Saved raw scores: {scores_path}")

    # Write thresholds to JSON
    out_path = OUTPUT_DIR / 'model_thresholds.json'
    with open(out_path, 'w') as f:
        json.dump(thresholds_out, f, indent=2)

    print(f"[OK] Thresholds saved: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
