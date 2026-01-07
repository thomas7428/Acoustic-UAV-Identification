"""
Modern Threshold Calibration
Analyse systématique des thresholds de décision à partir des résultats JSON précalculés.

Features:
- Utilise les fichiers JSON de config.PERFORMANCE_DIR
- Analyse l'impact des thresholds sur les métriques
- Génère des courbes ROC-like et threshold optimization
- Recommandations de thresholds optimaux par modèle
- Pas de recalcul de features, utilise les résultats existants
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Import project config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from tools import plot_utils

# Apply plotting style
plot_utils.set_style()

# Output directory
OUTPUT_DIR = plot_utils.get_output_dir(__file__)


def load_all_threshold_results():
    """
    Charge tous les résultats JSON et groupe par (model, split).
    """
    perf_dir = config.PERFORMANCE_DIR
    
    if not perf_dir.exists():
        print(f"[ERROR] Performance directory not found: {perf_dir}")
        return {}
    
    json_files = list(perf_dir.glob("*.json"))
    
    if not json_files:
        print(f"[ERROR] No JSON files found in {perf_dir}")
        return {}
    
    # Grouper par (model, split)
    grouped = defaultdict(list)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            model = metadata.get('model', 'UNKNOWN')
            split = metadata.get('split', 'UNKNOWN')
            threshold = metadata.get('threshold', 0.5)
            
            key = (model, split)
            grouped[key].append((threshold, data))
            
        except Exception as e:
            print(f"  ✗ Error loading {json_file.name}: {e}")
    
    # Trier par threshold
    for key in grouped:
        grouped[key].sort(key=lambda x: x[0])
    
    return dict(grouped)


def plot_threshold_curves(model, split, threshold_data, output_dir):
    """
    Plot les courbes de métriques vs threshold pour un modèle/split.
    """
    thresholds = [t for t, _ in threshold_data]
    
    # Extraire métriques
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    specificities = []
    fps = []
    fns = []
    
    for t, data in threshold_data:
        metrics = data.get('global_metrics', {})
        accuracies.append(metrics.get('accuracy', 0))
        precisions.append(metrics.get('precision', 0))
        recalls.append(metrics.get('recall', 0))
        f1_scores.append(metrics.get('f1_score', 0))
        specificities.append(metrics.get('specificity', 0))
        fps.append(metrics.get('fp', 0))
        fns.append(metrics.get('fn', 0))
    
    # Figure avec 3 subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Subplot 1: Métriques principales
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(thresholds, accuracies, marker='o', label='Accuracy', linewidth=2, markersize=8)
    ax1.plot(thresholds, precisions, marker='s', label='Precision', linewidth=2, markersize=8)
    ax1.plot(thresholds, recalls, marker='^', label='Recall', linewidth=2, markersize=8)
    ax1.plot(thresholds, f1_scores, marker='d', label='F1-Score', linewidth=2, markersize=8)
    ax1.plot(thresholds, specificities, marker='v', label='Specificity', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Decision Threshold', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax1.set_title(f'Metrics vs Threshold\n{model} | {split}', fontweight='bold', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # Marquer le meilleur F1
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = thresholds[best_f1_idx]
    best_f1_value = f1_scores[best_f1_idx]
    ax1.axvline(best_f1_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'Best F1 @ {best_f1_threshold:.2f}')
    ax1.plot(best_f1_threshold, best_f1_value, 'r*', markersize=20)
    
    # Subplot 2: FP vs FN trade-off
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(thresholds, fps, marker='o', label='False Positives', linewidth=2, markersize=8, color='#e74c3c')
    ax2.plot(thresholds, fns, marker='s', label='False Negatives', linewidth=2, markersize=8, color='#3498db')
    
    ax2.set_xlabel('Decision Threshold', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Count', fontweight='bold', fontsize=12)
    ax2.set_title(f'FP vs FN Trade-off\n{model} | {split}', fontweight='bold', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(best_f1_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Subplot 3: Precision-Recall trade-off
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(recalls, precisions, marker='o', linewidth=2, markersize=8, color='#9b59b6')
    
    # Annoter quelques thresholds clés
    for i in range(0, len(thresholds), max(1, len(thresholds)//5)):
        ax3.annotate(f't={thresholds[i]:.2f}', 
                    (recalls[i], precisions[i]),
                    textcoords="offset points", xytext=(5,-5), 
                    fontsize=8, alpha=0.7)
    
    ax3.set_xlabel('Recall (Sensitivity)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Precision', fontweight='bold', fontsize=12)
    ax3.set_title(f'Precision-Recall Curve\n{model} | {split}', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(0, 1.0)
    
    # Marquer le point optimal F1
    ax3.plot(recalls[best_f1_idx], precisions[best_f1_idx], 'r*', markersize=20, 
             label=f'Best F1 @ t={best_f1_threshold:.2f}')
    ax3.legend(fontsize=10)
    
    # Subplot 4: Recommandations
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Trouver quelques thresholds clés
    best_acc_idx = np.argmax(accuracies)
    best_prec_idx = np.argmax(precisions)
    best_rec_idx = np.argmax(recalls)
    
    # Trouver threshold équilibré (FP ≈ FN)
    fp_fn_diffs = [abs(fp - fn) for fp, fn in zip(fps, fns)]
    balanced_idx = np.argmin(fp_fn_diffs)
    
    recommendations = f"""
THRESHOLD RECOMMENDATIONS
{model} - {split} Split

Best F1-Score:
  Threshold: {thresholds[best_f1_idx]:.2f}
  F1: {f1_scores[best_f1_idx]:.4f}
  Precision: {precisions[best_f1_idx]:.4f}
  Recall: {recalls[best_f1_idx]:.4f}
  FP: {int(fps[best_f1_idx])}, FN: {int(fns[best_f1_idx])}

Best Accuracy:
  Threshold: {thresholds[best_acc_idx]:.2f}
  Accuracy: {accuracies[best_acc_idx]:.4f}
  FP: {int(fps[best_acc_idx])}, FN: {int(fns[best_acc_idx])}

Best Precision:
  Threshold: {thresholds[best_prec_idx]:.2f}
  Precision: {precisions[best_prec_idx]:.4f}
  Recall: {recalls[best_prec_idx]:.4f}

Best Recall:
  Threshold: {thresholds[best_rec_idx]:.2f}
  Recall: {recalls[best_rec_idx]:.4f}
  Precision: {precisions[best_rec_idx]:.4f}

Balanced (FP ≈ FN):
  Threshold: {thresholds[balanced_idx]:.2f}
  FP: {int(fps[balanced_idx])}, FN: {int(fns[balanced_idx])}
  Accuracy: {accuracies[balanced_idx]:.4f}
"""
    
    ax4.text(0.05, 0.95, recommendations, 
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    output_path = output_dir / f'threshold_calibration_{model}_{split}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()
    
    return {
        'best_f1_threshold': thresholds[best_f1_idx],
        'best_f1_value': f1_scores[best_f1_idx],
        'best_acc_threshold': thresholds[best_acc_idx],
        'balanced_threshold': thresholds[balanced_idx]
    }


def generate_recommendations_report(all_recommendations, output_dir):
    """
    Génère un rapport avec toutes les recommandations de thresholds.
    """
    report_path = output_dir / 'threshold_recommendations.json'
    
    # Convertir les clés tuple en strings pour JSON
    json_recommendations = {f"{model}_{split}": recs 
                           for (model, split), recs in all_recommendations.items()}
    
    with open(report_path, 'w') as f:
        json.dump(json_recommendations, f, indent=2)
    
    print(f"  ✓ Saved: {report_path}")
    
    # Rapport texte
    txt_path = output_dir / 'threshold_recommendations.txt'
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("THRESHOLD CALIBRATION RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        for (model, split), recs in all_recommendations.items():
            f.write(f"\n{model} - {split} Split\n")
            f.write(f"{'-'*80}\n")
            f.write(f"  Best F1 Threshold:       {recs['best_f1_threshold']:.2f} (F1={recs['best_f1_value']:.4f})\n")
            f.write(f"  Best Accuracy Threshold: {recs['best_acc_threshold']:.2f}\n")
            f.write(f"  Balanced Threshold:      {recs['balanced_threshold']:.2f} (FP≈FN)\n")
    
    print(f"  ✓ Saved: {txt_path}")


def main():
    """Point d'entrée principal."""
    print("\n" + "="*80)
    print("THRESHOLD CALIBRATION ANALYSIS")
    print("="*80 + "\n")
    
    print("[1/3] Loading threshold results from JSON files...")
    grouped_results = load_all_threshold_results()
    
    if not grouped_results:
        print("\n[ERROR] No multi-threshold results found!")
        print("Run Universal_Perf_Tester.py with multiple thresholds first.")
        return
    
    print(f"  Found {len(grouped_results)} model/split combinations")
    
    # Filtrer ceux qui ont au moins 3 thresholds
    multi_threshold = {k: v for k, v in grouped_results.items() if len(v) >= 3}
    
    if not multi_threshold:
        print("\n[WARNING] No results with 3+ thresholds found!")
        print("At least 3 different thresholds are needed for calibration analysis.")
        return
    
    print(f"  {len(multi_threshold)} with 3+ thresholds (usable for calibration)")
    
    print("\n[2/3] Generating threshold calibration plots...")
    all_recommendations = {}
    
    for (model, split), threshold_data in multi_threshold.items():
        print(f"  Processing {model} {split} ({len(threshold_data)} thresholds)...")
        recs = plot_threshold_curves(model, split, threshold_data, OUTPUT_DIR)
        all_recommendations[(model, split)] = recs
    
    print("\n[3/3] Generating recommendations report...")
    generate_recommendations_report(all_recommendations, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print(f"✓ Threshold calibration complete! Results in: {OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
