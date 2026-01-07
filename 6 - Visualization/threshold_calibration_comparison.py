#!/usr/bin/env python3
"""
Threshold Calibration Comparison (Style de l'ancien plot)
Crée un graphique à 4 subplots montrant la calibration des thresholds pour tous les modèles.
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_all_threshold_results():
    """
    Charge tous les résultats groupés par (model, split).
    Retourne: dict[(model, split)] = list[(threshold, metrics)]
    """
    perf_dir = config.PERFORMANCE_DIR
    json_files = list(perf_dir.glob("*_t*.json"))
    
    grouped = defaultdict(list)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            metrics = data.get('global_metrics', {})
            
            model = metadata.get('model', 'UNKNOWN')
            split = metadata.get('split', 'UNKNOWN')
            threshold = metadata.get('threshold', 0.5)
            
            grouped[(model, split)].append((threshold, metrics))
            
        except Exception as e:
            print(f"  ✗ Error loading {json_file.name}: {e}")
    
    # Trier chaque groupe par threshold
    for key in grouped:
        grouped[key].sort(key=lambda x: x[0])
    
    return grouped


def plot_threshold_calibration(grouped_results, output_file):
    """
    Crée le graphique à 4 subplots - TEST SET UNIQUEMENT pour clarté.
    """
    # Configuration du style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Threshold Calibration Comparison (Test Set Only)', fontsize=16, fontweight='bold')
    
    # Couleurs par modèle
    colors = {
        'CNN': '#1f77b4',
        'RNN': '#ff7f0e',
        'CRNN': '#2ca02c',
        'ATTENTION_CRNN': '#d62728'
    }
    
    # FILTRER: garder seulement les résultats du test set
    test_results = {k: v for k, v in grouped_results.items() if k[1] == 'test'}
    
    # Préparer les données pour chaque subplot
    for (model, split), results in sorted(test_results.items()):
        if not results:
            continue
        
        thresholds = [r[0] for r in results]
        metrics = [r[1] for r in results]
        
        # Extraire les métriques
        f1_scores = [m.get('f1_score', 0) for m in metrics]
        precisions = [m.get('precision', 0) for m in metrics]
        recalls = [m.get('recall', 0) for m in metrics]
        accuracies = [m.get('accuracy', 0) for m in metrics]
        specificities = [m.get('specificity', 0) for m in metrics]
        
        color = colors.get(model, '#333333')
        label = f"{model}"
        
        # Subplot 1: F1-Score vs Threshold
        axes[0, 0].plot(thresholds, f1_scores, marker='o', label=label, color=color, linewidth=2.5, markersize=6)
        
        # Subplot 2: Precision & Recall vs Threshold (combiné sur même subplot)
        axes[0, 1].plot(thresholds, precisions, marker='s', label=f"{label} (Precision)", 
                       color=color, linewidth=2, linestyle='--', markersize=5, alpha=0.7)
        axes[0, 1].plot(thresholds, recalls, marker='^', label=f"{label} (Recall)", 
                       color=color, linewidth=2, linestyle='-', markersize=5)
        
        # Subplot 3: Accuracy vs Threshold
        axes[1, 0].plot(thresholds, accuracies, marker='D', label=label, color=color, linewidth=2.5, markersize=6)
        
        # Subplot 4: Specificity vs Threshold
        axes[1, 1].plot(thresholds, specificities, marker='o', label=label, 
                       color=color, linewidth=2.5, markersize=6)
        axes[1, 1].plot(thresholds, specificities, marker='x', label=f"{label} (Spec)", 
                       color=color, linewidth=2, linestyle='--', markersize=4)
        
        # Trouver le meilleur threshold (F1 max) et ajouter une ligne verticale
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        
        # Ligne verticale rouge au meilleur threshold
        for ax in axes.flat:
            ax.axvline(best_threshold, color=color, alpha=0.3, linestyle='--', linewidth=1)
    
    # Configuration des axes
    axes[0, 0].set_xlabel('Threshold', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('F1-Score vs Threshold', fontsize=13, fontweight='bold')
    axes[0, 0].legend(loc='best', fontsize=10, framealpha=0.9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])
    
    axes[0, 1].set_xlabel('Threshold', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Precision & Recall vs Threshold', fontsize=13, fontweight='bold')
    axes[0, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, framealpha=0.9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    axes[1, 0].set_xlabel('Threshold', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Accuracy vs Threshold', fontsize=13, fontweight='bold')
    axes[1, 0].legend(loc='best', fontsize=10, framealpha=0.9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.05])
    
    axes[1, 1].set_xlabel('Threshold', fontsize=12)
    axes[1, 1].set_ylabel('Specificity', fontsize=12)
    axes[1, 1].set_title('Specificity vs Threshold', fontsize=13, fontweight='bold')
    axes[1, 1].legend(loc='best', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def main():
    print("=" * 80)
    print("  THRESHOLD CALIBRATION COMPARISON")
    print("=" * 80)
    print()
    
    # Charger tous les résultats avec thresholds
    print("[INFO] Loading threshold results...")
    grouped = load_all_threshold_results()
    
    if not grouped:
        print("[ERROR] No threshold results found!")
        return
    
    print(f"[INFO] Found {len(grouped)} (model, split) configurations")
    
    # Créer le graphique
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "threshold_calibration_comparison.png"
    
    print("[INFO] Creating calibration plot...")
    plot_threshold_calibration(grouped, output_file)
    
    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
