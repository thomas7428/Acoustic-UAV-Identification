"""
Threshold Calibration for Model Predictions
============================================

Tests different classification thresholds to find optimal balance
between precision and recall for drone detection.

Current issue: Models predict DRONE everywhere (high recall, low precision)
Solution: Increase threshold from 0.5 to 0.6-0.8 to require higher confidence
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import librosa
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

# Import project config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Import ML libraries
from tensorflow import keras

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# Paths
DATASET_TEST_PATH = Path(__file__).parent.parent / "0 - DADS dataset extraction" / "dataset_test"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Audio parameters
SAMPLE_RATE = 22050
DURATION = 4.0


def load_models():
    """Load trained models."""
    models = {}
    
    model_paths = {
        'CNN': config.CNN_MODEL_PATH,
        'RNN': config.RNN_MODEL_PATH,
        'CRNN': config.CRNN_MODEL_PATH,
        'Attention-CRNN': config.ATTENTION_CRNN_MODEL_PATH
    }
    
    for name, path in model_paths.items():
        if path.exists():
            try:
                models[name] = keras.models.load_model(path, compile=False)
                print(f"[OK] Loaded {name}")
            except Exception as e:
                print(f"[WARNING] Failed to load {name}: {e}")
        else:
            print(f"[WARNING] Model not found: {path}")
    
    return models


def extract_mel_features(audio_path):
    """Extract MEL spectrogram matching training parameters."""
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
    
    # MEL spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=44, n_fft=2048, hop_length=512
    )
    
    # Convert to dB
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Pad or trim to 90 time steps
    if mel_db.shape[1] < 90:
        mel_db = np.pad(mel_db, ((0, 0), (0, 90 - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :90]
    
    return mel_db


def load_test_data(max_samples_per_class=500):
    """Load test data from dataset_test."""
    X_test, y_test = [], []
    
    for class_label in [0, 1]:
        class_dir = DATASET_TEST_PATH / str(class_label)
        class_name = "no-drone" if class_label == 0 else "drone"
        
        if not class_dir.exists():
            print(f"[WARNING] Directory not found: {class_dir}")
            continue
        
        wav_files = list(class_dir.glob("*.wav"))[:max_samples_per_class]
        
        print(f"[INFO] Loading {len(wav_files)} {class_name} samples...")
        
        for wav_file in wav_files:
            try:
                mel = extract_mel_features(wav_file)
                X_test.append(mel)
                y_test.append(class_label)
            except Exception as e:
                print(f"[WARNING] Failed to process {wav_file.name}: {e}")
                continue
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\n[OK] Loaded {len(X_test)} test samples")
    print(f"  - No-drone: {np.sum(y_test == 0)}")
    print(f"  - Drone:    {np.sum(y_test == 1)}")
    
    return X_test, y_test


def get_predictions_probabilities(model, model_name, X_test):
    """Get prediction probabilities for all test samples."""
    # Reshape for model input
    if model_name in ['CNN', 'CRNN', 'Attention-CRNN']:
        X_input = X_test[..., np.newaxis]  # Add channel dimension
    else:  # RNN
        X_input = X_test
    
    # Get probabilities
    probs = model.predict(X_input, verbose=0)
    
    # Extract drone class probability (class 1)
    drone_probs = probs[:, 1]
    
    return drone_probs


def evaluate_threshold(y_true, drone_probs, threshold):
    """Evaluate performance at a specific threshold."""
    y_pred = (drone_probs >= threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }


def calibrate_model(model_name, model, X_test, y_test):
    """Calibrate threshold for a single model."""
    print(f"\n{'='*60}")
    print(f"Calibrating {model_name}")
    print('='*60)
    
    # Get prediction probabilities
    drone_probs = get_predictions_probabilities(model, model_name, X_test)
    
    # Test different thresholds
    thresholds = np.arange(0.1, 1.0, 0.05)
    results = []
    
    for threshold in thresholds:
        result = evaluate_threshold(y_test, drone_probs, threshold)
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Find optimal thresholds
    best_f1_idx = results_df['f1'].idxmax()
    best_balanced_idx = (results_df['recall'] + results_df['specificity']).idxmax()
    
    best_f1 = results_df.iloc[best_f1_idx]
    best_balanced = results_df.iloc[best_balanced_idx]
    
    print(f"\n[BASELINE] Threshold = 0.50")
    baseline = evaluate_threshold(y_test, drone_probs, 0.50)
    print(f"  Accuracy:    {baseline['accuracy']*100:.1f}%")
    print(f"  Precision:   {baseline['precision']*100:.1f}%")
    print(f"  Recall:      {baseline['recall']*100:.1f}%")
    print(f"  Specificity: {baseline['specificity']*100:.1f}%")
    print(f"  F1-Score:    {baseline['f1']*100:.1f}%")
    
    print(f"\n[OPTIMAL F1] Threshold = {best_f1['threshold']:.2f}")
    print(f"  Accuracy:    {best_f1['accuracy']*100:.1f}%")
    print(f"  Precision:   {best_f1['precision']*100:.1f}%")
    print(f"  Recall:      {best_f1['recall']*100:.1f}%")
    print(f"  Specificity: {best_f1['specificity']*100:.1f}%")
    print(f"  F1-Score:    {best_f1['f1']*100:.1f}%")
    
    print(f"\n[BALANCED] Threshold = {best_balanced['threshold']:.2f}")
    print(f"  Accuracy:    {best_balanced['accuracy']*100:.1f}%")
    print(f"  Precision:   {best_balanced['precision']*100:.1f}%")
    print(f"  Recall:      {best_balanced['recall']*100:.1f}%")
    print(f"  Specificity: {best_balanced['specificity']*100:.1f}%")
    print(f"  F1-Score:    {best_balanced['f1']*100:.1f}%")
    
    return results_df, drone_probs


def plot_calibration_results(all_results):
    """Plot threshold calibration results for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'CNN': 'steelblue', 'RNN': 'coral', 'CRNN': 'mediumseagreen', 'Attention-CRNN': 'purple'}
    
    for model_name, results_df in all_results.items():
        color = colors.get(model_name, 'gray')
        
        # Plot 1: F1-Score vs Threshold
        ax = axes[0, 0]
        ax.plot(results_df['threshold'], results_df['f1'], 
                label=model_name, linewidth=2, color=color, marker='o', markersize=4)
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.set_title('F1-Score vs Classification Threshold', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')
        
        # Plot 2: Precision/Recall vs Threshold
        ax = axes[0, 1]
        ax.plot(results_df['threshold'], results_df['precision'], 
                label=f'{model_name} Precision', linestyle='-', linewidth=2, color=color)
        ax.plot(results_df['threshold'], results_df['recall'], 
                label=f'{model_name} Recall', linestyle='--', linewidth=2, color=color)
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Precision/Recall vs Threshold', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.5)
        
        # Plot 3: Accuracy vs Threshold
        ax = axes[1, 0]
        ax.plot(results_df['threshold'], results_df['accuracy'], 
                label=model_name, linewidth=2, color=color, marker='o', markersize=4)
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy vs Threshold', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.5)
        
        # Plot 4: Recall vs Specificity (Balance)
        ax = axes[1, 1]
        ax.plot(results_df['threshold'], results_df['recall'], 
                label=f'{model_name} Recall', linestyle='-', linewidth=2, color=color)
        ax.plot(results_df['threshold'], results_df['specificity'], 
                label=f'{model_name} Specificity', linestyle='--', linewidth=2, color=color)
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Recall vs Specificity (Balance)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'threshold_calibration.png', dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved: {OUTPUT_DIR / 'threshold_calibration.png'}")
    
    return fig


def generate_calibration_report(all_results):
    """Generate summary report with optimal thresholds."""
    report_data = []
    
    for model_name, results_df in all_results.items():
        # Find optimal thresholds
        best_f1_idx = results_df['f1'].idxmax()
        best_balanced_idx = (results_df['recall'] + results_df['specificity']).idxmax()
        
        # Find closest to 0.50 threshold (might be 0.45 or 0.55)
        baseline_idx = (results_df['threshold'] - 0.50).abs().idxmin()
        baseline = results_df.iloc[baseline_idx]
        best_f1 = results_df.iloc[best_f1_idx]
        best_balanced = results_df.iloc[best_balanced_idx]
        
        report_data.append({
            'Model': model_name,
            'Baseline Threshold': f"{baseline['threshold']:.2f}",
            'Baseline F1': f"{baseline['f1']*100:.1f}%",
            'Baseline Precision': f"{baseline['precision']*100:.1f}%",
            'Baseline Recall': f"{baseline['recall']*100:.1f}%",
            'Optimal F1 Threshold': f"{best_f1['threshold']:.2f}",
            'Optimal F1 Score': f"{best_f1['f1']*100:.1f}%",
            'Balanced Threshold': f"{best_balanced['threshold']:.2f}",
            'Balanced F1': f"{best_balanced['f1']*100:.1f}%"
        })
    
    report_df = pd.DataFrame(report_data)
    
    # Save to CSV
    report_df.to_csv(OUTPUT_DIR / 'threshold_calibration_report.csv', index=False)
    print(f"[OK] Saved: {OUTPUT_DIR / 'threshold_calibration_report.csv'}")
    
    # Print summary
    print("\n" + "="*80)
    print("THRESHOLD CALIBRATION SUMMARY")
    print("="*80)
    print(report_df.to_string(index=False))
    print("="*80)
    
    return report_df


def main():
    print("\n" + "="*80)
    print("THRESHOLD CALIBRATION FOR DRONE DETECTION")
    print("="*80)
    print("\nObjective: Find optimal classification threshold to balance")
    print("           precision (avoid false alarms) and recall (detect all drones)")
    print("\nCurrent issue: Models predict DRONE everywhere (high FP rate)")
    print("Solution: Increase threshold to require higher confidence")
    print("="*80)
    
    # Load models
    print("\n[1/4] Loading models...")
    models = load_models()
    
    if not models:
        print("[ERROR] No models loaded!")
        return
    
    # Load test data
    print("\n[2/4] Loading test data...")
    X_test, y_test = load_test_data(max_samples_per_class=500)
    
    # Calibrate each model
    print("\n[3/4] Calibrating thresholds...")
    all_results = {}
    all_probs = {}
    
    for model_name, model in models.items():
        results_df, drone_probs = calibrate_model(model_name, model, X_test, y_test)
        all_results[model_name] = results_df
        all_probs[model_name] = drone_probs
    
    # Generate visualizations
    print("\n[4/4] Generating visualizations...")
    plot_calibration_results(all_results)
    
    # Generate report
    generate_calibration_report(all_results)
    
    print("\n[DONE] Threshold calibration complete!")
    print(f"Check results in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
