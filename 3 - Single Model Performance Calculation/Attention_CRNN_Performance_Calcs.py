"""
Performance Calculator for Attention-Enhanced CRNN Model

Evaluates the attention-enhanced model on test data.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# Paths
DATASET_TEST = "../0 - DADS dataset extraction/dataset_test"
MODEL_PATH = "../0 - DADS dataset extraction/saved_models/attention_crnn_model.keras"
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


def main():
    print("\n" + "="*70)
    print("  Attention-Enhanced CRNN - Performance Evaluation")
    print("="*70)
    
    # Check model exists
    if not Path(MODEL_PATH).exists():
        print(f"\n[ERROR] Model not found: {MODEL_PATH}")
        print("[INFO] Please train the Attention-Enhanced CRNN first:")
        print("       cd '2 - Model Training' && python Attention_CRNN_Trainer.py")
        return
    
    # Load model
    print(f"\n[INFO] Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    
    # Load test features
    print(f"[INFO] Loading test data from {DATASET_TEST}...")
    features_path = Path("../0 - DADS dataset extraction/extracted_features/mel_pitch_shift_9.0.json")
    
    with open(features_path, 'r') as f:
        data = json.load(f)
    
    X_test = np.array(data['X_test'])
    y_test = np.array(data['y_test'])
    
    # Add channel dimension
    X_test = X_test[..., np.newaxis]
    
    print(f"[INFO] Test samples: {len(X_test)}")
    
    # Make predictions
    print("\n[INFO] Making predictions...")
    predictions = model.predict(X_test, verbose=0)
    
    # Extract class 1 probabilities
    y_pred_prob = predictions[:, 1]
    y_true = y_test
    
    # Calculate metrics
    scores = calculate_metrics(y_true, y_pred_prob)
    
    # Display results
    print("\n" + "="*70)
    print("  Overall Performance")
    print("="*70)
    print(f"  Accuracy:  {scores['Accuracy']*100:.2f}%")
    print(f"  Precision: {scores['Precision']*100:.2f}%")
    print(f"  Recall:    {scores['Recall']*100:.2f}%")
    print(f"  F1 Score:  {scores['F1 Score']*100:.2f}%")
    print(f"  TP: {int(scores['TP'])} | FN: {int(scores['FN'])} | TN: {int(scores['TN'])} | FP: {int(scores['FP'])}")
    print("="*70)
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    scores_file = Path(RESULTS_DIR) / "attention_crnn_scores.json"
    with open(scores_file, 'w') as f:
        json.dump({k: [v] for k, v in scores.items()}, f, indent=4)
    
    print(f"\n[SUCCESS] Scores saved to {scores_file}")
    
    # Save predictions
    predictions_data = []
    for i in range(len(y_test)):
        predictions_data.append({
            "true_label": int(y_test[i]),
            "predicted_prob": float(y_pred_prob[i]),
            "predicted_label": int(y_pred_prob[i] >= 0.5)
        })
    
    preds_file = Path(RESULTS_DIR) / "attention_crnn_predictions.json"
    with open(preds_file, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    print(f"[SUCCESS] Predictions saved to {preds_file}")
    print("\n[SUCCESS] Performance evaluation complete!")


if __name__ == "__main__":
    main()
