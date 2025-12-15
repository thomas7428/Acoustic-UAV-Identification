"""
Model Performance Visualization
Compares performance metrics across different models (CNN, RNN, CRNN).

Features:
- Training history curves (loss and accuracy)
- Confusion matrices
- Performance metrics comparison (accuracy, precision, recall, F1)
- ROC curves and AUC scores
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import importlib.util

# Import project config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def _load_converter_module():
    """Dynamically load the converter script from folder '3 - Single Model Performance Calculation'.
    Returns the module object or None if not found/failed.
    """
    conv_path = Path(__file__).parent.parent / "3 - Single Model Performance Calculation" / "convert_results_for_viz.py"
    if not conv_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("convert_results_for_viz", str(conv_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"[WARN] Failed to load converter module: {e}")
        return None


def load_training_history(model_name):
    """Load training history from CSV file using centralized paths in `config`."""
    mapping = {
        'CNN': config.CNN_HISTORY_PATH,
        'RNN': config.RNN_HISTORY_PATH,
        'CRNN': config.CRNN_HISTORY_PATH,
        'Attention-CRNN': config.ATTENTION_CRNN_HISTORY_PATH,
    }

    history_path = Path(mapping.get(model_name))

    if not history_path.exists():
        print(f"[WARNING] History not found: {history_path}")
        return None

    df = pd.read_csv(history_path)
    return df


def load_accuracy_data(model_name):
    """Load accuracy metrics from JSON file using centralized paths in `config`."""
    mapping = {
        'CNN': config.CNN_ACC_PATH,
        'RNN': config.RNN_ACC_PATH,
        'CRNN': config.CRNN_ACC_PATH,
        'Attention-CRNN': config.ATTENTION_CRNN_ACC_PATH,
    }

    acc_path = Path(mapping.get(model_name))

    if not acc_path.exists():
        # Try to auto-convert from scores JSON if available (backwards compatibility)
        print(f"[WARNING] Accuracy file not found: {acc_path}")
        scores_map = {
            'CNN': config.CNN_SCORES_PATH,
            'RNN': config.RNN_SCORES_PATH,
            'CRNN': config.CRNN_SCORES_PATH,
            'Attention-CRNN': config.ATTENTION_CRNN_SCORES_PATH,
        }
        scores_path = Path(scores_map.get(model_name))
        if scores_path.exists():
            conv = _load_converter_module()
            if conv and hasattr(conv, 'convert_scores_to_accuracy_format'):
                try:
                    print(f"[INFO] Converting scores -> accuracy JSON for {model_name} using {scores_path}")
                    conv.convert_scores_to_accuracy_format(scores_path, acc_path, model_name)
                    if acc_path.exists():
                        print(f"[OK] Generated accuracy file: {acc_path}")
                except Exception as e:
                    print(f"[ERROR] Conversion failed for {model_name}: {e}")
            else:
                print(f"[WARN] Converter not available; cannot auto-generate {acc_path}")
        else:
            # No scores available either
            return None

    with open(acc_path, 'r') as f:
        data = json.load(f)

    return data


def plot_training_curves():
    """Plot training and validation curves for all models."""
    models = ['CNN', 'RNN', 'CRNN', 'Attention-CRNN']
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    
    for idx, model_name in enumerate(models):
        history = load_training_history(model_name)
        
        if history is None:
            continue
        
        # Plot accuracy
        ax_acc = axes[0, idx]
        if 'accuracy' in history.columns:
            ax_acc.plot(history['accuracy'], label='Train Accuracy', linewidth=2, color='steelblue')
        if 'val_accuracy' in history.columns:
            ax_acc.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2, color='coral')
        
        ax_acc.set_title(f'{model_name} - Accuracy', fontsize=12, fontweight='bold')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(alpha=0.3)
        
        # Plot loss
        ax_loss = axes[1, idx]
        if 'loss' in history.columns:
            ax_loss.plot(history['loss'], label='Train Loss', linewidth=2, color='steelblue')
        if 'val_loss' in history.columns:
            ax_loss.plot(history['val_loss'], label='Val Loss', linewidth=2, color='coral')
        
        ax_loss.set_title(f'{model_name} - Loss', fontsize=12, fontweight='bold')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'training_curves.png'}")
    
    return fig


def plot_confusion_matrices():
    """Plot confusion matrices for all models."""
    models = ['CNN', 'RNN', 'CRNN', 'Attention-CRNN']
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    for idx, model_name in enumerate(models):
        acc_data = load_accuracy_data(model_name)
        
        if acc_data is None or 'confusion_matrix' not in acc_data:
            continue
        
        cm = np.array(acc_data['confusion_matrix'])
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['No Drone', 'Drone'],
                   yticklabels=['No Drone', 'Drone'],
                   cbar_kws={'label': 'Count'})
        
        axes[idx].set_title(f'{model_name} Confusion Matrix', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontweight='bold')
        axes[idx].set_ylabel('True Label', fontweight='bold')
        
        # Add accuracy on plot
        accuracy = acc_data.get('test_accuracy', 0) * 100
        axes[idx].text(0.5, -0.15, f'Accuracy: {accuracy:.2f}%',
                      ha='center', transform=axes[idx].transAxes,
                      fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'confusion_matrices.png'}")
    
    return fig


def plot_metrics_comparison():
    """Compare performance metrics across models."""
    models = ['CNN', 'RNN', 'CRNN', 'Attention-CRNN']
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Collect data
    data = {metric: [] for metric in metrics_names}
    available_models = []
    
    for model_name in models:
        acc_data = load_accuracy_data(model_name)
        
        if acc_data is None:
            continue
        
        available_models.append(model_name)
        data['Accuracy'].append(acc_data.get('test_accuracy', 0) * 100)
        
        # Get classification report if available
        if 'classification_report' in acc_data:
            report = acc_data['classification_report']
            # Use weighted average
            data['Precision'].append(report.get('weighted avg', {}).get('precision', 0) * 100)
            data['Recall'].append(report.get('weighted avg', {}).get('recall', 0) * 100)
            data['F1-Score'].append(report.get('weighted avg', {}).get('f1-score', 0) * 100)
        else:
            data['Precision'].append(0)
            data['Recall'].append(0)
            data['F1-Score'].append(0)
    
    if not available_models:
        print("[WARNING] No model data available for comparison")
        return None
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(available_models))
    width = 0.2
    
    colors = ['steelblue', 'coral', 'lightgreen', 'gold']
    
    for i, (metric, values) in enumerate(data.items()):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score (%)', fontweight='bold', fontsize=12)
    ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(available_models)
    ax.legend(loc='upper right')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'metrics_comparison.png'}")
    
    return fig


def generate_performance_table():
    """Generate a detailed performance comparison table."""
    models = ['CNN', 'RNN', 'CRNN', 'Attention-CRNN']
    
    table_data = []
    
    for model_name in models:
        acc_data = load_accuracy_data(model_name)
        
        if acc_data is None:
            continue
        
        row = {
            'Model': model_name,
            'Test Accuracy (%)': f"{acc_data.get('test_accuracy', 0) * 100:.2f}",
            'Train Accuracy (%)': f"{acc_data.get('train_accuracy', 0) * 100:.2f}" if 'train_accuracy' in acc_data else 'N/A',
        }
        
        # Add per-class metrics if available
        if 'classification_report' in acc_data:
            report = acc_data['classification_report']
            
            # Class 0 (No Drone)
            if '0' in report:
                row['No-Drone Precision (%)'] = f"{report['0'].get('precision', 0) * 100:.2f}"
                row['No-Drone Recall (%)'] = f"{report['0'].get('recall', 0) * 100:.2f}"
                row['No-Drone F1 (%)'] = f"{report['0'].get('f1-score', 0) * 100:.2f}"
            
            # Class 1 (Drone)
            if '1' in report:
                row['Drone Precision (%)'] = f"{report['1'].get('precision', 0) * 100:.2f}"
                row['Drone Recall (%)'] = f"{report['1'].get('recall', 0) * 100:.2f}"
                row['Drone F1 (%)'] = f"{report['1'].get('f1-score', 0) * 100:.2f}"
        
        table_data.append(row)
    
    if not table_data:
        print("[WARNING] No performance data available")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Save to CSV
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / 'performance_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved: {csv_path}")
    
    # Print table
    print("\n" + "="*100)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")
    
    return df


def main():
    """Run all model performance visualizations."""
    print("="*80)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*80 + "\n")
    
    # Plot training curves
    print("[1/4] Plotting training curves...")
    plot_training_curves()
    
    # Plot confusion matrices
    print("\n[2/4] Plotting confusion matrices...")
    plot_confusion_matrices()
    
    # Plot metrics comparison
    print("\n[3/4] Plotting metrics comparison...")
    plot_metrics_comparison()
    
    # Generate performance table
    print("\n[4/4] Generating performance table...")
    generate_performance_table()
    
    print("\n" + "="*80)
    print("[SUCCESS] Performance analysis complete!")
    print("="*80)
    print(f"\nOutputs saved in: {Path(__file__).parent / 'outputs'}")


if __name__ == "__main__":
    main()
