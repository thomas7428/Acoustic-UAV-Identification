#!/usr/bin/env python3
"""
Threshold Analysis Visualization

Crée des visualisations pour analyser les thresholds calibrés:
- Courbes de métriques vs threshold
- Points optimaux sélectionnés
- Comparaison multi-modèles
- Pareto fronts (precision vs recall)
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_calibration_data():
    """Charge les données de calibration."""
    if not config.CALIBRATION_FILE_PATH.exists():
        raise FileNotFoundError(f"Calibration file not found: {config.CALIBRATION_FILE_PATH}")
    
    with open(config.CALIBRATION_FILE_PATH) as f:
        return json.load(f)


def plot_threshold_curves_single_model(model_name, model_data, output_dir):
    """Crée un plot détaillé pour un seul modèle."""
    all_results = model_data.get('all_tested_thresholds', [])
    if not all_results:
        print(f"No threshold data for {model_name}")
        return
    
    # Extract data
    thresholds = [r['threshold'] for r in all_results]
    f1_scores = [r['f1_score'] for r in all_results]
    precisions_drone = [r['precision_drone'] for r in all_results]
    precisions_ambient = [r['precision_ambient'] for r in all_results]
    recalls = [r['recall'] for r in all_results]
    balanced_precisions = [r['balanced_precision'] for r in all_results]
    
    optimal_threshold = model_data['threshold']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} - Threshold Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: F1-Score vs Threshold
    ax = axes[0, 0]
    ax.plot(thresholds, f1_scores, 'o-', color='#2ecc71', linewidth=2, markersize=4, label='F1-Score')
    ax.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, label=f'Optimal: {optimal_threshold:.3f}')
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('F1-Score', fontsize=11)
    ax.set_title('F1-Score vs Threshold', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Precisions vs Threshold
    ax = axes[0, 1]
    ax.plot(thresholds, precisions_drone, 'o-', color='#3498db', linewidth=2, markersize=4, label='Precision Drone (PPV)')
    ax.plot(thresholds, precisions_ambient, 's-', color='#e74c3c', linewidth=2, markersize=4, label='Precision Ambient (NPV)')
    ax.plot(thresholds, balanced_precisions, '^-', color='#9b59b6', linewidth=2, markersize=4, label='Balanced Precision')
    ax.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precisions vs Threshold', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Recall/Specificity vs Threshold
    ax = axes[1, 0]
    specificities = [r['specificity'] for r in all_results]
    ax.plot(thresholds, recalls, 'o-', color='#e67e22', linewidth=2, markersize=4, label='Recall (Sensitivity)')
    ax.plot(thresholds, specificities, 's-', color='#1abc9c', linewidth=2, markersize=4, label='Specificity')
    ax.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Rate', fontsize=11)
    ax.set_title('Recall/Specificity vs Threshold', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Pareto Front (Precision vs Recall)
    ax = axes[1, 1]
    
    # Color gradient by threshold value
    scatter = ax.scatter(recalls, precisions_drone, c=thresholds, cmap='viridis', 
                        s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    
    # Mark optimal point
    optimal_metrics = model_data['metrics_at_threshold']
    ax.scatter([optimal_metrics['recall']], [optimal_metrics['precision_drone']], 
              color='red', s=200, marker='*', edgecolors='black', linewidths=1.5,
              label=f'Optimal (t={optimal_threshold:.3f})', zorder=10)
    
    # Add constraint lines if available
    constraints = model_data.get('constraints_met', {})
    for constraint_name, is_met in constraints.items():
        if 'min_recall' in constraint_name:
            min_val = float(constraint_name.split('_')[-1])
            ax.axvline(min_val, color='orange', linestyle=':', alpha=0.5, label=f'Min Recall: {min_val:.2f}')
        elif 'min_precision_drone' in constraint_name:
            min_val = float(constraint_name.split('_')[-1])
            ax.axhline(min_val, color='blue', linestyle=':', alpha=0.5, label=f'Min Precision: {min_val:.2f}')
    
    ax.set_xlabel('Recall (Sensitivity)', fontsize=11)
    ax.set_ylabel('Precision Drone (PPV)', fontsize=11)
    ax.set_title('Precision-Recall Trade-off (Pareto Front)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Threshold', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'threshold_analysis_{model_name.lower()}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path.name}")


def plot_threshold_comparison_all_models(calib_data, output_dir):
    """Compare thresholds across all models."""
    models = calib_data.get('models', {})
    
    if not models:
        print("No model data to compare")
        return
    
    # Extract data
    model_names = []
    thresholds = []
    f1_scores = []
    balanced_precisions = []
    recalls = []
    
    for model_name, model_data in models.items():
        if 'error' in model_data:
            continue
        
        model_names.append(model_name)
        thresholds.append(model_data['threshold'])
        
        metrics = model_data['metrics_at_threshold']
        f1_scores.append(metrics['f1_score'])
        balanced_precisions.append(metrics['balanced_precision'])
        recalls.append(metrics['recall'])
    
    if not model_names:
        print("No valid model data for comparison")
        return
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Threshold Calibration - Model Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # Plot 1: Thresholds
    ax = axes[0, 0]
    bars = ax.bar(model_names, thresholds, color=colors[:len(model_names)], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Optimal Threshold', fontsize=11)
    ax.set_title('Calibrated Thresholds by Model', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: F1-Scores
    ax = axes[0, 1]
    bars = ax.bar(model_names, f1_scores, color=colors[:len(model_names)], alpha=0.7, edgecolor='black')
    ax.set_ylabel('F1-Score', fontsize=11)
    ax.set_title('F1-Score at Optimal Threshold', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Balanced Precision
    ax = axes[1, 0]
    bars = ax.bar(model_names, balanced_precisions, color=colors[:len(model_names)], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Balanced Precision', fontsize=11)
    ax.set_title('Balanced Precision (min(PPV, NPV))', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Recall
    ax = axes[1, 1]
    bars = ax.bar(model_names, recalls, color=colors[:len(model_names)], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Recall', fontsize=11)
    ax.set_title('Recall (Sensitivity) at Optimal Threshold', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add constraint line
    constraints = calib_data.get('constraints', {})
    if 'min_recall' in constraints:
        ax.axhline(constraints['min_recall'], color='red', linestyle='--', 
                  linewidth=2, label=f"Min Recall: {constraints['min_recall']:.2f}")
        ax.legend()
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'threshold_comparison_all_models.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path.name}")


def main():
    """Generate threshold analysis visualizations."""
    print("="*80)
    print("  THRESHOLD ANALYSIS VISUALIZATION")
    print("="*80)
    
    # Load calibration data
    try:
        calib_data = load_calibration_data()
        print(f"\n✓ Loaded calibration data from {config.CALIBRATION_FILE_PATH}")
    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        print("\nRun calibrate_thresholds.py first to generate calibration data.")
        return 1
    
    # Output directory
    output_dir = config.VIZ_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating threshold analysis plots...")
    print(f"Output: {output_dir}")
    print("")
    
    # Generate plots for each model
    models = calib_data.get('models', {})
    for model_name, model_data in models.items():
        if 'error' in model_data:
            print(f"✗ Skipping {model_name} (calibration failed)")
            continue
        
        plot_threshold_curves_single_model(model_name, model_data, output_dir)
    
    # Generate comparison plot
    plot_threshold_comparison_all_models(calib_data, output_dir)
    
    print("")
    print("="*80)
    print("  THRESHOLD ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nVisualizations saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
