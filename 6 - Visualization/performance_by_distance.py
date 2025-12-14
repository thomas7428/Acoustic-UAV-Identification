"""
Performance by Distance Analysis - Dynamic Configuration Version
Analyzes REAL model performance as a function of simulated drone distance (SNR categories).

This script:
1. Dynamically loads augmentation categories from augment_config_v2.json
2. Tests trained models on each category separately
3. Generates real performance metrics by distance/SNR
4. Adapts to any future configuration changes

Features:
- Automatic category detection from config
- Real accuracy by SNR category
- Performance degradation analysis
- Per-model robustness comparison
- Confusion matrices by distance
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import defaultdict
import librosa
import math
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', message='n_fft=.*is too large for input signal of length=.*')

# Import project config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Import ML libraries
try:
    from tensorflow import keras
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
except ImportError:
    print("[ERROR] TensorFlow/scikit-learn not installed")
    sys.exit(1)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Audio processing parameters
SAMPLE_RATE = 22050
DURATION = 10
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Paths
AUGMENT_CONFIG_PATH = Path(__file__).parent.parent / "0 - DADS dataset extraction" / "augment_config_v2.json"
DATASET_COMBINED_PATH = Path(__file__).parent.parent / "0 - DADS dataset extraction" / "dataset_combined"
FEATURES_JSON_PATH = Path(__file__).parent.parent / "0 - DADS dataset extraction" / "extracted_features" / "mel_pitch_shift_9.0.json"
OUTPUT_DIR = Path(__file__).parent / "outputs"


def load_augmentation_config():
    """Load augmentation configuration to get dynamic categories."""
    if not AUGMENT_CONFIG_PATH.exists():
        print(f"[ERROR] Config not found: {AUGMENT_CONFIG_PATH}")
        return None
    
    try:
        with open(AUGMENT_CONFIG_PATH, 'r') as f:
            config_data = json.load(f)
        
        print(f"[OK] Loaded augmentation config v{config_data.get('version', 'unknown')}")
        return config_data
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return None


def get_categories_from_config(aug_config):
    """
    Extract category definitions from augmentation config.
    
    Returns dict with category info:
    {
        'drone_500m': {
            'name': 'drone_500m',
            'display_name': '500m (-32dB)',
            'snr_db': -32,
            'proportion': 0.40,
            'description': '...',
            'label': 1
        },
        ...
    }
    """
    if not aug_config:
        return {}
    
    categories = {}
    
    # Extract drone augmentation categories
    if aug_config.get('drone_augmentation', {}).get('enabled'):
        for cat in aug_config['drone_augmentation']['categories']:
            name = cat['name']
            snr = cat['snr_db']
            
            # Parse distance from name (e.g., 'drone_500m' -> '500m')
            distance = name.replace('drone_', '').replace('_', ' ')
            
            categories[name] = {
                'name': name,
                'display_name': f"{distance} ({snr:+d}dB)".replace('+-', '-'),
                'snr_db': snr,
                'proportion': cat.get('proportion', 0),
                'description': cat.get('description', ''),
                'label': cat.get('label', 1),
                'file_pattern': f"aug_{name}_"
            }
    
    # Extract no-drone augmentation categories (if any)
    if aug_config.get('no_drone_augmentation', {}).get('enabled'):
        for cat in aug_config['no_drone_augmentation']['categories']:
            name = cat['name']
            snr = cat.get('snr_db', 0)
            
            categories[name] = {
                'name': name,
                'display_name': cat.get('display_name', name),
                'snr_db': snr,
                'proportion': cat.get('proportion', 0),
                'description': cat.get('description', ''),
                'label': cat.get('label', 0),
                'file_pattern': f"aug_{name}_"
            }
    
    # Add original clean samples category
    categories['original_clean'] = {
        'name': 'original_clean',
        'display_name': 'Original (Clean)',
        'snr_db': float('inf'),  # Perfect quality
        'proportion': 0,
        'description': 'Original unaugmented samples',
        'label': None,  # Will contain both classes
        'file_pattern': 'orig_'
    }
    
    return categories


def get_category_files(categories):
    """
    Get all audio files for each category from dataset_combined.
    
    Returns:
    {
        'category_name': {
            'files': [Path objects],
            'display_name': 'Display Name',
            'snr_db': -32,
            'expected_label': 0 or 1 or None
        }
    }
    """
    if not DATASET_COMBINED_PATH.exists():
        print(f"[ERROR] Dataset not found: {DATASET_COMBINED_PATH}")
        return {}
    
    category_files = {}
    
    for cat_name, cat_info in categories.items():
        pattern = cat_info['file_pattern']
        files = []
        
        # Search in both class folders (0 and 1)
        for class_dir in ['0', '1']:
            class_path = DATASET_COMBINED_PATH / class_dir
            if class_path.exists():
                # Find files matching pattern
                matching_files = list(class_path.glob(f"{pattern}*.wav"))
                files.extend(matching_files)
        
        if files:
            category_files[cat_name] = {
                'files': sorted(files),
                'display_name': cat_info['display_name'],
                'snr_db': cat_info['snr_db'],
                'expected_label': cat_info['label']
            }
    
    return category_files


def load_precomputed_features():
    """Load pre-computed MEL features from JSON file."""
    if not FEATURES_JSON_PATH.exists():
        print(f"[ERROR] Features JSON not found: {FEATURES_JSON_PATH}")
        return None
    
    try:
        print(f"[INFO] Loading pre-computed features from {FEATURES_JSON_PATH}...")
        with open(FEATURES_JSON_PATH, 'r') as f:
            data = json.load(f)
        
        mel_features = np.array(data['mel'])  # Shape: (48000, 44, 90)
        labels = np.array(data['labels'])
        
        print(f"[OK] Loaded {len(mel_features)} pre-computed MEL features")
        print(f"[INFO] Feature shape: {mel_features[0].shape}")
        
        return {'features': mel_features, 'labels': labels}
    except Exception as e:
        print(f"[ERROR] Failed to load features: {e}")
        return None


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
                print(f"[OK] Loaded {name} model")
            except Exception as e:
                print(f"[WARNING] Failed to load {name}: {e}")
        else:
            print(f"[WARNING] Model not found: {path}")
    
    return models


def get_precomputed_features_for_file(file_path, precomputed_data, num_segments=10):
    """Get MEL features using EXACT same parameters as training.
    
    CRITICAL: Training uses 4 seconds with shape (44, 90).
    Test MUST match this exactly or we get catastrophic failure.
    
    Args:
        file_path: Path to audio file
        precomputed_data: Not used (kept for compatibility)
        num_segments: Ignored - we use single 4-second window like training
    
    Returns:
        Array of shape (44, 90) - SINGLE MEL spectrogram matching training
    """
    n_mels = 44  # MATCH training (not 90!)
    n_fft = 2048
    hop_length = 512
    
    # Load audio - EXACTLY 4 seconds like training
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=4.0)
    
    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(y=audio, 
                                        sr=sr,
                                        n_fft=n_fft,
                                        hop_length=hop_length,
                                        n_mels=n_mels)
    
    # Convert to dB
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Pad or trim to exactly 90 time steps (matching training)
    if mel_db.shape[1] < 90:
        mel_db = np.pad(mel_db, ((0, 0), (0, 90 - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :90]
    
    # Return shape (44, 90) - NO segments, single prediction like training
    return mel_db


def evaluate_model_on_category(model, model_name, files, precomputed_data, feature_type='mel'):
    """
    Evaluate a model on files from a specific category.
    
    Args:
        model: Trained model
        model_name: Name of model (CNN, RNN, CRNN)
        files: List of Path objects to audio files
        precomputed_data: Pre-computed features dict (not used, kept for compatibility)
        feature_type: 'mel' or 'mfcc'
    
    Returns:
        Dictionary with accuracy, precision, recall, f1 metrics
    """
    print(f"  Evaluating {len(files)} files... ", end='', flush=True)
    
    predictions = []
    labels = []
    
    for file_path in files:
        try:
            # Get true label from parent directory name
            true_label = int(file_path.parent.name)
            
            # Get features - shape (44, 90) matching training exactly
            if feature_type == 'mel':
                features = get_precomputed_features_for_file(file_path, precomputed_data)
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")
            
            # Reshape for model input - add batch and channel dimensions
            # Training shape: (batch, 44, 90, 1)
            # Test must match: (1, 44, 90, 1)
            if model_name in ['CNN', 'CRNN', 'Attention-CRNN']:
                features = features[np.newaxis, ..., np.newaxis]  # (44, 90) -> (1, 44, 90, 1)
            else:  # RNN
                features = features[np.newaxis, ...]  # (44, 90) -> (1, 44, 90)
            
            # Predict - single prediction, no averaging
            pred = model.predict(features, verbose=0)
            
            # Extract prediction from batch dimension
            pred = pred[0]  # (1, 2) -> (2)
            
            # CALIBRATED THRESHOLDS (from threshold calibration analysis)
            # Models require higher confidence to predict DRONE (reduce FP rate)
            optimal_thresholds = {
                'CNN': 0.85,           # Baseline: 92.1% F1 → Optimal: 94.9% F1
                'RNN': 0.95,           # Baseline: 76.2% F1 → Optimal: 90.2% F1 (biggest gain)
                'CRNN': 0.90,          # Baseline: 87.4% F1 → Optimal: 93.4% F1
                'Attention-CRNN': 0.95 # Baseline: 88.1% F1 → Optimal: 94.1% F1
            }
            
            threshold = optimal_thresholds.get(model_name, 0.5)
            drone_prob = pred[1]  # Probability of class 1 (DRONE)
            
            # Apply calibrated threshold
            pred_label = 1 if drone_prob >= threshold else 0
            
            # Debug first few predictions
            if len(predictions) < 3:
                print(f"\n  [DEBUG] File: {file_path.name}, True label: {true_label}")
                print(f"  [DEBUG] Features shape: {features.shape}, drone_prob: {drone_prob:.3f}, threshold: {threshold:.2f}, pred_label: {pred_label}")
            
            predictions.append(pred_label)
            labels.append(true_label)
            
        except Exception as e:
            print(f"\n[WARNING] Error processing {file_path.name}: {e}")
            continue
    
    if not predictions:
        print("No valid predictions")
        return None
    
    # Debug: Show prediction distribution
    pred_array = np.array(predictions)
    label_array = np.array(labels)
    print(f"\n  [DEBUG] Predictions: {np.sum(pred_array == 0)} no-drones, {np.sum(pred_array == 1)} drones")
    print(f"  [DEBUG] True labels: {np.sum(label_array == 0)} no-drones, {np.sum(label_array == 1)} drones")
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Handle cases where category has only one class
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        # Single class - use simple accuracy
        precision = accuracy
        recall = accuracy
        f1 = accuracy
    else:
        # Binary classification
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, 
                                                                    average='binary', 
                                                                    zero_division=0)
    
    print(f"  Accuracy: {accuracy*100:.1f}%, Recall: {recall*100:.1f}%, Precision: {precision*100:.1f}%")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
        'labels': labels
    }


def test_models_by_distance(categories, category_files):
    """Test all models on each distance category."""
    print("[1/3] Loading pre-computed features...")
    precomputed_data = load_precomputed_features()
    
    if precomputed_data is None:
        print("[ERROR] Failed to load pre-computed features")
        return None
    
    print("\n[2/3] Loading models...")
    models = load_models()
    
    if not models:
        print("[ERROR] No models loaded")
        return None
    
    print(f"\n[3/3] Testing on {len(category_files)} categories...")
    
    # Test each model on each category
    results = defaultdict(dict)
    
    for model_name, model in models.items():
        print(f"\nTesting {model_name}:")
        print("-" * 60)
        
        # All models use mel spectrograms in this project
        feature_type = 'mel'
        
        for cat_name, cat_data in category_files.items():
            files = cat_data['files']
            display_name = cat_data['display_name']
            
            print(f"{display_name:25s} ({len(files):4d} files): ", end='')
            
            # Limit to 100 files for speed
            result = evaluate_model_on_category(model, model_name, files[:100], precomputed_data, feature_type)
            
            if result:
                results[model_name][cat_name] = {
                    **result,
                    'display_name': display_name,
                    'snr_db': cat_data['snr_db']
                }
    
    return results


def generate_performance_table(results, categories):
    """Generate detailed performance table by distance."""
    if not results:
        return None
    
    # Sort categories by SNR (worst to best)
    sorted_cats = sorted(
        [(cat_name, cat_info) for cat_name, cat_info in categories.items()],
        key=lambda x: x[1]['snr_db']
    )
    
    table_data = []
    
    for cat_name, cat_info in sorted_cats:
        # Check if this category has results
        has_results = any(cat_name in model_results for model_results in results.values())
        if not has_results:
            continue
        
        row = {'Category': cat_info['display_name']}
        
        for model_name, model_results in results.items():
            if cat_name in model_results:
                res = model_results[cat_name]
                row[f'{model_name} Acc (%)'] = f"{res['accuracy']*100:.2f}"
                row[f'{model_name} F1 (%)'] = f"{res['f1']*100:.2f}"
        
        if len(row) > 1:  # Has data beyond category name
            table_data.append(row)
    
    if not table_data:
        return None
    
    df = pd.DataFrame(table_data)
    
    # Save to CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "performance_by_distance.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved: {csv_path}")
    
    # Print formatted table
    print("\n" + "=" * 100)
    print("REAL PERFORMANCE BY DISTANCE/SNR")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    
    return df


def plot_performance_by_distance(results, categories):
    """Plot real model performance by distance category."""
    if not results:
        print("[WARNING] No results to plot")
        return None
    
    # Sort categories by SNR (worst to best)
    sorted_cats = sorted(
        [(cat_name, cat_info) for cat_name, cat_info in categories.items()],
        key=lambda x: x[1]['snr_db']
    )
    
    # Filter to categories with results
    cat_order = [
        (cat_name, cat_info) 
        for cat_name, cat_info in sorted_cats 
        if any(cat_name in model_results for model_results in results.values())
    ]
    
    if not cat_order:
        print("[WARNING] No categories with results")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract display names and SNRs
    display_names = [cat_info['display_name'] for _, cat_info in cat_order]
    cat_names = [cat_name for cat_name, _ in cat_order]
    snr_values = [cat_info['snr_db'] for _, cat_info in cat_order]
    
    # Plot 1: Accuracy by category
    ax1 = axes[0, 0]
    x = np.arange(len(cat_names))
    width = 0.25
    
    colors = {'CNN': 'steelblue', 'RNN': 'coral', 'CRNN': 'lightgreen'}
    
    for i, (model_name, model_results) in enumerate(results.items()):
        accuracies = [
            model_results[cat_name]['accuracy'] * 100 if cat_name in model_results else 0
            for cat_name in cat_names
        ]
        ax1.bar(x + i*width, accuracies, width, label=model_name, color=colors.get(model_name, 'gray'))
    
    ax1.set_xlabel('Distance Category')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy by Drone Distance', fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(display_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Plot 2: F1-Score by category
    ax2 = axes[0, 1]
    
    for i, (model_name, model_results) in enumerate(results.items()):
        f1_scores = [
            model_results[cat_name]['f1'] * 100 if cat_name in model_results else 0
            for cat_name in cat_names
        ]
        ax2.bar(x + i*width, f1_scores, width, label=model_name, color=colors.get(model_name, 'gray'))
    
    ax2.set_xlabel('Distance Category')
    ax2.set_ylabel('F1-Score (%)')
    ax2.set_title('Model F1-Score by Drone Distance', fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(display_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 105])
    
    # Plot 3: Performance vs SNR (line plot)
    ax3 = axes[1, 0]
    
    # Filter out infinite SNR (original clean)
    finite_snr_indices = [i for i, snr in enumerate(snr_values) if np.isfinite(snr)]
    
    if finite_snr_indices:
        finite_snrs = [snr_values[i] for i in finite_snr_indices]
        finite_cats = [cat_names[i] for i in finite_snr_indices]
        
        for model_name, model_results in results.items():
            accuracies = [
                model_results[cat_name]['accuracy'] * 100 if cat_name in model_results else 0
                for cat_name in finite_cats
            ]
            ax3.plot(finite_snrs, accuracies, marker='o', linewidth=2, 
                    label=model_name, color=colors.get(model_name, 'gray'))
    
    ax3.set_xlabel('SNR (dB)')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Performance vs. Drone Distance (SNR)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 105])
    
    # Plot 4: Performance degradation
    ax4 = axes[1, 1]
    
    if finite_snr_indices and len(finite_snr_indices) > 1:
        # Calculate degradation from best to worst SNR
        for model_name, model_results in results.items():
            accuracies = [
                model_results[cat_name]['accuracy'] * 100 if cat_name in model_results else 0
                for cat_name in finite_cats
            ]
            
            # Degradation relative to best performance
            best_acc = max(accuracies) if accuracies else 0
            degradations = [best_acc - acc for acc in accuracies]
            
            ax4.plot(finite_snrs, degradations, marker='s', linewidth=2,
                    label=model_name, color=colors.get(model_name, 'gray'))
    
    ax4.set_xlabel('SNR (dB)')
    ax4.set_ylabel('Performance Degradation (%)')
    ax4.set_title('Model Robustness to Noise (Lower is Better)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "real_performance_by_distance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved: {output_path}")
    
    plt.close()
    return output_path


def plot_difficulty_spectrum(categories, category_files):
    """Plot the difficulty spectrum showing SNR levels and sample counts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sort categories by SNR
    sorted_cats = sorted(
        [(cat_name, cat_info) for cat_name, cat_info in categories.items()],
        key=lambda x: x[1]['snr_db']
    )
    
    # Filter to categories with files
    filtered_cats = [
        (cat_name, cat_info) 
        for cat_name, cat_info in sorted_cats 
        if cat_name in category_files
    ]
    
    if not filtered_cats:
        print("[WARNING] No categories to plot")
        return
    
    display_names = [cat_info['display_name'] for _, cat_info in filtered_cats]
    cat_names = [cat_name for cat_name, _ in filtered_cats]
    snr_values = [cat_info['snr_db'] if np.isfinite(cat_info['snr_db']) else 0 for _, cat_info in filtered_cats]
    file_counts = [len(category_files[cat_name]['files']) for cat_name in cat_names]
    
    # Plot 1: SNR levels
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(cat_names)))
    
    bars1 = ax1.bar(display_names, snr_values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax1.set_ylabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax1.set_title('Difficulty Spectrum: SNR by Category', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='SNR = 0dB')
    ax1.legend()
    
    # Add value labels on bars
    for bar, snr in zip(bars1, snr_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{snr:+.0f}dB' if np.isfinite(snr) else 'Clean',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Sample distribution
    bars2 = ax2.bar(display_names, file_counts, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax2.set_title('Dataset Distribution by Category', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars2, file_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "difficulty_spectrum.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    
    plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("REAL PERFORMANCE BY DISTANCE ANALYSIS")
    print("=" * 80)
    print("\nTesting trained models on each SNR category...")
    print("This will take several minutes...\n")
    
    # Load configuration
    print("[STEP 1/5] Loading augmentation configuration...")
    aug_config = load_augmentation_config()
    if not aug_config:
        print("[ERROR] Cannot proceed without configuration")
        return
    
    # Extract categories from config
    print("\n[STEP 2/5] Extracting category definitions...")
    categories = get_categories_from_config(aug_config)
    print(f"[OK] Found {len(categories)} categories:")
    for cat_name, cat_info in categories.items():
        print(f"  - {cat_info['display_name']}: {cat_info.get('description', 'N/A')[:60]}")
    
    # Get files for each category
    print(f"\n[STEP 3/5] Scanning dataset files...")
    category_files = get_category_files(categories)
    print(f"[OK] Found files for {len(category_files)} categories:")
    for cat_name, cat_data in category_files.items():
        print(f"  - {cat_data['display_name']}: {len(cat_data['files'])} files")
    
    if not category_files:
        print("[ERROR] No category files found!")
        return
    
    # Test models
    print(f"\n[STEP 4/5] Testing models by distance category...")
    results = test_models_by_distance(categories, category_files)
    
    if not results:
        print("[ERROR] No results generated")
        return
    
    # Generate outputs
    print(f"\n[STEP 5/5] Generating performance analysis...")
    
    print("\n[5a/5] Generating performance table...")
    generate_performance_table(results, categories)
    
    print("\n[5b/5] Plotting performance by distance...")
    plot_performance_by_distance(results, categories)
    
    print("\n[5c/5] Plotting difficulty spectrum...")
    plot_difficulty_spectrum(categories, category_files)
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Real performance by distance analysis complete!")
    print("=" * 80)
    print(f"\nOutputs saved in: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
