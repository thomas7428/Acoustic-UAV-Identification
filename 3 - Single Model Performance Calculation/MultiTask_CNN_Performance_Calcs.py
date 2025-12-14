"""
Performance Calculator for Multi-Task CNN Model

Evaluates the multi-task model on test data and generates predictions.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from training script
from MultiTask_CNN_Trainer import load_data_with_metadata

# Paths
DATASET_TEST = "../0 - DADS dataset extraction/dataset_test"
MODEL_PATH = "../0 - DADS dataset extraction/saved_models/multitask_cnn_model.keras"
RESULTS_DIR = "../0 - DADS dataset extraction/results/predictions"


def calculate_metrics(y_true, y_pred):
    """Calculate TP, FN, TN, FP and derived metrics."""
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    TP = np.sum((y_true == 1) & (y_pred_binary == 1))
    FN = np.sum((y_true == 1) & (y_pred_binary == 0))
    TN = np.sum((y_true == 0) & (y_pred_binary == 0))
    FP = np.sum((y_true == 0) & (y_pred_binary == 1))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "TP": float(TP),
        "FN": float(FN),
        "TN": float(TN),
        "FP": float(FP),
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1 Score": float(f1)
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  Multi-Task CNN - Performance Evaluation")
    print("="*70 + "\n")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("[INFO] Please train the Multi-Task CNN first:")
        print("       cd '2 - Model Training' && python MultiTask_CNN_Trainer.py")
        sys.exit(1)
    
    # Load model
    print(f"[INFO] Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    
    # Load test data
    print(f"[INFO] Loading test data from {DATASET_TEST}...")
    X_test, y_class_test, y_dist_test, y_snr_test, filenames = load_data_with_metadata(DATASET_TEST)
    
    print(f"[INFO] Test samples: {len(X_test)}")
    print()
    
    # Predict
    print("[INFO] Making predictions...")
    predictions = model.predict(X_test, verbose=0)
    
    # Extract detection predictions (first output)
    detection_probs = predictions[0]  # [n_samples, 2]
    distance_probs = predictions[1]   # [n_samples, 6]
    snr_probs = predictions[2]        # [n_samples, 5]
    
    # Get probability of drone class (index 1)
    y_pred_probs = detection_probs[:, 1]
    
    # Calculate overall metrics
    metrics = calculate_metrics(y_class_test, y_pred_probs)
    
    print("\n" + "="*70)
    print("  Overall Performance")
    print("="*70)
    print(f"  Accuracy:  {metrics['Accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['Precision']*100:.2f}%")
    print(f"  Recall:    {metrics['Recall']*100:.2f}%")
    print(f"  F1 Score:  {metrics['F1 Score']*100:.2f}%")
    print(f"  TP: {metrics['TP']:.0f} | FN: {metrics['FN']:.0f} | TN: {metrics['TN']:.0f} | FP: {metrics['FP']:.0f}")
    print("="*70 + "\n")
    
    # Save scores
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    scores_file = os.path.join(RESULTS_DIR, "multitask_cnn_scores.json")
    with open(scores_file, 'w') as f:
        json.dump({k: [v] for k, v in metrics.items()}, f, indent=4)
    print(f"[SUCCESS] Scores saved to {scores_file}")
    
    # Save detailed predictions
    predictions_data = []
    for i, filename in enumerate(filenames):
        predictions_data.append({
            "filename": filename,
            "true_class": int(y_class_test[i]),
            "pred_prob_drone": float(y_pred_probs[i]),
            "pred_class": int(y_pred_probs[i] >= 0.5),
            "true_distance": int(y_dist_test[i]),
            "pred_distance": int(np.argmax(distance_probs[i])),
            "true_snr": int(y_snr_test[i]),
            "pred_snr": int(np.argmax(snr_probs[i]))
        })
    
    predictions_file = os.path.join(RESULTS_DIR, "multitask_cnn_predictions.json")
    with open(predictions_file, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    print(f"[SUCCESS] Predictions saved to {predictions_file}")
    
    # Analyze distance prediction accuracy
    print("\n" + "="*70)
    print("  Auxiliary Task Performance")
    print("="*70)
    
    distance_correct = np.sum(np.argmax(distance_probs, axis=1) == y_dist_test)
    distance_acc = distance_correct / len(y_dist_test) * 100
    print(f"  Distance Estimation Accuracy: {distance_acc:.2f}%")
    
    snr_correct = np.sum(np.argmax(snr_probs, axis=1) == y_snr_test)
    snr_acc = snr_correct / len(y_snr_test) * 100
    print(f"  SNR Estimation Accuracy: {snr_acc:.2f}%")
    print("="*70 + "\n")
    
    print("[SUCCESS] Performance evaluation complete!")
