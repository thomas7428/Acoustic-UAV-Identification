#!/usr/bin/env python3
"""
Automatic Threshold Calibration for Model Predictions

This script automatically determines optimal classification thresholds for each model
to achieve target recall while maximizing precision. Particularly useful for the RNN
model which shows systematic bias toward the negative class.

Usage:
    python calibrate_thresholds.py [--target-recall 0.95] [--save]
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import tensorflow as tf

# Import project config
sys.path.insert(0, str(Path(__file__).parent))
import config

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_validation_data():
    """Load validation data from MEL features JSON."""
    print("[INFO] Loading validation data...")
    
    with open(config.MEL_TRAIN_PATH, 'r') as f:
        data = json.load(f)
    
    mel_features = np.array(data['mel'])
    labels = np.array(data['labels'])
    
    # Split: assuming 80% train, 10% val, 10% test
    # Use validation set (indices 80-90%)
    n_samples = len(labels)
    val_start = int(0.8 * n_samples)
    val_end = int(0.9 * n_samples)
    
    X_val = mel_features[val_start:val_end]
    y_val = labels[val_start:val_end]
    
    print(f"[OK] Loaded {len(X_val)} validation samples")
    return X_val, y_val


def get_model_predictions(model_path, X_val, model_name):
    """Get probability predictions from a model."""
    print(f"\n[INFO] Loading {model_name} model...")
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"[OK] Model loaded: {model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None
    
    # Reshape for model input
    if model_name == "RNN":
        # RNN expects (batch, timesteps, features)
        X_input = X_val
    else:
        # CNN/CRNN expect (batch, timesteps, features, channels)
        X_input = X_val[..., np.newaxis]
    
    print(f"[INFO] Predicting on {len(X_val)} samples...")
    predictions = model.predict(X_input, verbose=0, batch_size=32)
    
    # Extract probability for positive class (drone = index 1)
    probs_positive = predictions[:, 1]
    
    print(f"[OK] Predictions complete")
    print(f"  Min prob: {probs_positive.min():.4f}")
    print(f"  Max prob: {probs_positive.max():.4f}")
    print(f"  Mean prob: {probs_positive.mean():.4f}")
    print(f"  Std prob: {probs_positive.std():.4f}")
    
    return probs_positive


def calibrate_threshold(y_true, y_probs, target_recall=0.95, model_name="Model"):
    """
    Find optimal threshold to achieve target recall.
    
    Args:
        y_true: True labels (0 or 1)
        y_probs: Predicted probabilities for positive class
        target_recall: Target recall to achieve (default: 0.95)
        model_name: Name of model for display
    
    Returns:
        dict with optimal_threshold and metrics
    """
    print(f"\n{'='*60}")
    print(f"CALIBRATING THRESHOLD FOR {model_name}")
    print(f"{'='*60}")
    print(f"Target recall: {target_recall:.1%}")
    
    # Test thresholds from 0.01 to 0.99
    thresholds = np.arange(0.01, 1.0, 0.01)
    metrics = []
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        
        # Calculate metrics
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        metrics.append({
            'threshold': thresh,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'accuracy': accuracy,
            'distance_to_target': abs(recall - target_recall)
        })
    
    # Find threshold closest to target recall
    metrics_sorted = sorted(metrics, key=lambda x: x['distance_to_target'])
    optimal = metrics_sorted[0]
    
    # Also find threshold with best F1 score
    metrics_by_f1 = sorted(metrics, key=lambda x: x['f1'], reverse=True)
    best_f1 = metrics_by_f1[0]
    
    print(f"\n[RESULT] Optimal Threshold (target recall {target_recall:.1%}):")
    print(f"  Threshold:  {optimal['threshold']:.3f}")
    print(f"  Recall:     {optimal['recall']:.3f}")
    print(f"  Precision:  {optimal['precision']:.3f}")
    print(f"  F1 Score:   {optimal['f1']:.3f}")
    print(f"  Accuracy:   {optimal['accuracy']:.3f}")
    
    print(f"\n[INFO] Alternative: Best F1 Score Threshold:")
    print(f"  Threshold:  {best_f1['threshold']:.3f}")
    print(f"  Recall:     {best_f1['recall']:.3f}")
    print(f"  Precision:  {best_f1['precision']:.3f}")
    print(f"  F1 Score:   {best_f1['f1']:.3f}")
    print(f"  Accuracy:   {best_f1['accuracy']:.3f}")
    
    # Default threshold performance for comparison
    default_pred = (y_probs >= 0.5).astype(int)
    default_recall = recall_score(y_true, default_pred, zero_division=0)
    default_precision = precision_score(y_true, default_pred, zero_division=0)
    default_f1 = f1_score(y_true, default_pred, zero_division=0)
    
    print(f"\n[COMPARISON] Default Threshold (0.500):")
    print(f"  Recall:     {default_recall:.3f}")
    print(f"  Precision:  {default_precision:.3f}")
    print(f"  F1 Score:   {default_f1:.3f}")
    
    improvement = (optimal['f1'] - default_f1) / default_f1 * 100 if default_f1 > 0 else float('inf')
    print(f"\n[GAIN] F1 Score improvement: {improvement:+.1f}%")
    
    return {
        'optimal_threshold': optimal['threshold'],
        'target_recall_threshold': optimal['threshold'],
        'best_f1_threshold': best_f1['threshold'],
        'metrics_at_optimal': {
            'recall': optimal['recall'],
            'precision': optimal['precision'],
            'f1': optimal['f1'],
            'accuracy': optimal['accuracy']
        },
        'metrics_at_best_f1': {
            'recall': best_f1['recall'],
            'precision': best_f1['precision'],
            'f1': best_f1['f1'],
            'accuracy': best_f1['accuracy']
        },
        'metrics_at_default': {
            'recall': default_recall,
            'precision': default_precision,
            'f1': default_f1
        }
    }


def calibrate_all_models(target_recall=0.95, save=False):
    """Calibrate thresholds for all models."""
    print("="*70)
    print("AUTOMATIC THRESHOLD CALIBRATION")
    print("="*70)
    
    # Load validation data once
    X_val, y_val = load_validation_data()
    
    # Models to calibrate
    models = {
        "CNN": config.CNN_MODEL_PATH,
        "RNN": config.RNN_MODEL_PATH,
        "CRNN": config.CRNN_MODEL_PATH
    }
    
    calibrated_thresholds = {}
    
    for model_name, model_path in models.items():
        if not model_path.exists():
            print(f"\n[WARNING] Model not found: {model_path}")
            continue
        
        # Get predictions
        probs = get_model_predictions(model_path, X_val, model_name)
        
        if probs is None:
            continue
        
        # Calibrate threshold
        result = calibrate_threshold(y_val, probs, target_recall, model_name)
        calibrated_thresholds[model_name] = result
    
    # Display summary
    print("\n" + "="*70)
    print("CALIBRATION SUMMARY")
    print("="*70)
    print(f"{'Model':<10} {'Default':<10} {'Optimal':<10} {'Recall':<10} {'Precision':<10} {'F1':<10}")
    print("-"*70)
    
    for model_name, result in calibrated_thresholds.items():
        default_f1 = result['metrics_at_default']['f1']
        optimal_f1 = result['metrics_at_optimal']['f1']
        recall = result['metrics_at_optimal']['recall']
        precision = result['metrics_at_optimal']['precision']
        threshold = result['optimal_threshold']
        
        print(f"{model_name:<10} {0.5:<10.3f} {threshold:<10.3f} {recall:<10.3f} {precision:<10.3f} {optimal_f1:<10.3f}")
    
    # Save results
    if save:
        output_path = config.RESULTS_DIR / "calibrated_thresholds.json"
        
        # Simplify for saving
        save_data = {
            model: {
                'threshold': result['optimal_threshold'],
                'best_f1_threshold': result['best_f1_threshold'],
                'metrics': result['metrics_at_optimal']
            }
            for model, result in calibrated_thresholds.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=4)
        
        print(f"\n[SAVED] Calibrated thresholds: {output_path}")
        
        # Also update config.py programmatically (append to file)
        config_update = f"""
# Auto-calibrated thresholds (generated on {Path(__file__).stem})
# To use these, set MODEL_THRESHOLDS in config.py or load from calibrated_thresholds.json
CALIBRATED_THRESHOLDS = {{
    "CNN": {save_data.get('CNN', {}).get('threshold', 0.5):.3f},
    "RNN": {save_data.get('RNN', {}).get('threshold', 0.5):.3f},
    "CRNN": {save_data.get('CRNN', {}).get('threshold', 0.5):.3f},
}}
"""
        print(f"\n[INFO] To apply these thresholds, add to config.py:")
        print(config_update)
    
    return calibrated_thresholds


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate optimal classification thresholds for models'
    )
    parser.add_argument(
        '--target-recall',
        type=float,
        default=0.95,
        help='Target recall to achieve (default: 0.95)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save calibrated thresholds to JSON file'
    )
    
    args = parser.parse_args()
    
    calibrate_all_models(target_recall=args.target_recall, save=args.save)
    
    print("\n[DONE] Calibration complete!")


if __name__ == "__main__":
    main()
