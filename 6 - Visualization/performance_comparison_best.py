"""
Performance Comparison - Best Results Only
Charge uniquement les meilleurs résultats (un threshold par modèle/split)
et génère des visualisations claires et utiles.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Debug: show calibration file and normalized thresholds
try:
    print(f"[DEBUG performance_comparison_best] CALIBRATION_FILE_PATH={getattr(config,'CALIBRATION_FILE_PATH',None)}")
    print(f"[DEBUG performance_comparison_best] MODEL_THRESHOLDS_NORMALIZED={config.MODEL_THRESHOLDS_NORMALIZED}")
except Exception:
    pass
from tools import plot_utils

plot_utils.set_style()
OUTPUT_DIR = plot_utils.get_output_dir(__file__)


def load_best_results():
    """Charge le résumé des meilleurs résultats."""
    summary_file = config.PERFORMANCE_DIR / "best_results_summary.json"
    
    if not summary_file.exists():
        print(f"[ERROR] Best results not found: {summary_file}")
        print("[INFO] Ensure canonical performance JSONs exist in config.PERFORMANCE_DIR and run the visualizer pipeline.")
        return None
    
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    # Charger les fichiers JSON complets correspondants
    results = {}
    perf_dir = config.PERFORMANCE_DIR
    
    for key, summary in data.get('results', {}).items():
        model = summary['model']
        split = summary['split']
        threshold = summary['threshold']
        
        # Trouver le fichier JSON correspondant (canonique: {model}_{split}.json)
        json_file = perf_dir / f"{model.lower()}_{split}.json"

        if not json_file.exists():
            print(f"  ✗ Missing canonical perf file: {json_file.name}")
            continue

        try:
            with open(json_file, 'r') as f:
                full_data = json.load(f)

            # Prefer threshold found in file metadata; fall back to summary threshold
            file_threshold = full_data.get('metadata', {}).get('threshold', threshold)
            key_tuple = (model, split, file_threshold)
            results[key_tuple] = full_data
            print(f"  ✓ Loaded: {model} | {split} | t={file_threshold:.2f}")

        except Exception as e:
            print(f"  ✗ Error loading {json_file.name}: {e}")
    
    return results


def plot_global_metrics_comparison(results, output_dir):
    """Comparaison des métriques globales - meilleurs résultats uniquement."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Global Performance Comparison (Best Thresholds)', fontsize=16, fontweight='bold')
    
    # Grouper par modèle et split
    model_split_data = defaultdict(lambda: defaultdict(dict))
    
    for (model, split, threshold), data in results.items():
        metrics = data.get('global_metrics', {})
        model_split_data[model][split] = metrics
    
    models = sorted(model_split_data.keys())
    splits = sorted({split for model_data in model_split_data.values() for split in model_data.keys()})
    
    metrics_to_plot = [
        ('accuracy', 'Accuracy', axes[0, 0]),
        ('f1_score', 'F1-Score', axes[0, 1]),
        ('precision', 'Precision', axes[1, 0]),
        ('recall', 'Recall', axes[1, 1])
    ]
    
    colors = {'train': '#2ecc71', 'val': '#3498db', 'test': '#e74c3c'}
    width = 0.25
    
    for metric_key, metric_name, ax in metrics_to_plot:
        x = np.arange(len(models))
        
        for i, split in enumerate(splits):
            values = [model_split_data[model].get(split, {}).get(metric_key, 0) for model in models]
            offset = (i - len(splits)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=split.capitalize(), 
                         color=colors.get(split, '#95a5a6'), alpha=0.8, edgecolor='black')
            
            # Ajouter les valeurs sur les barres
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    output_path = output_dir / 'best_global_metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_confusion_matrices(results, output_dir):
    """Matrices de confusion - test set uniquement."""
    test_results = {k: v for k, v in results.items() if k[1] == 'test'}
    
    if not test_results:
        print("  [SKIP] No test results found")
        return
    
    n_models = len(test_results)
    cols = min(2, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(10*cols, 8*rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]
    
    fig.suptitle('Confusion Matrices (Test Set - Best Thresholds)', fontsize=16, fontweight='bold')
    
    for idx, ((model, split, threshold), data) in enumerate(sorted(test_results.items())):
        metrics = data.get('global_metrics', {})
        
        tp = metrics.get('tp', 0)
        tn = metrics.get('tn', 0)
        fp = metrics.get('fp', 0)
        fn = metrics.get('fn', 0)
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   cbar_kws={'label': 'Count'}, annot_kws={'size': 14, 'weight': 'bold'})
        
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('Actual', fontweight='bold')
        ax.set_title(f'{model} | t={threshold:.2f}\nAcc={metrics.get("accuracy", 0):.3f}, F1={metrics.get("f1_score", 0):.3f}', 
                    fontweight='bold')
        ax.set_xticklabels(['No Drone', 'Drone'])
        ax.set_yticklabels(['No Drone', 'Drone'])
    
    # Masquer axes inutilisés
    for idx in range(len(test_results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'best_confusion_matrices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_class_performance(results, output_dir):
    """Performance par classe (drone vs ambient) - test set."""
    test_results = {k: v for k, v in results.items() if k[1] == 'test'}
    
    if not test_results:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Class Performance (Test Set - Best Thresholds)', fontsize=16, fontweight='bold')
    
    models = sorted([k[0] for k in test_results.keys()])
    
    # Préparer les données
    drone_acc = []
    ambient_acc = []
    drone_f1 = []
    ambient_f1 = []
    
    for model in models:
        key = [k for k in test_results.keys() if k[0] == model][0]
        data = test_results[key]
        class_metrics = data.get('class_metrics', {})
        
        drone_metrics = class_metrics.get('drone', {})
        ambient_metrics = class_metrics.get('ambient', {})
        
        drone_acc.append(drone_metrics.get('accuracy', 0))
        ambient_acc.append(ambient_metrics.get('accuracy', 0))
        drone_f1.append(drone_metrics.get('f1_score', 0))
        ambient_f1.append(ambient_metrics.get('f1_score', 0))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Subplot 1: Accuracy
    axes[0].bar(x - width/2, drone_acc, width, label='Drone', color='#3498db', alpha=0.8, edgecolor='black')
    axes[0].bar(x + width/2, ambient_acc, width, label='Ambient', color='#2ecc71', alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('Model', fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontweight='bold')
    axes[0].set_title('Accuracy by Class', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0, 1.05)
    
    # Subplot 2: F1-Score
    axes[1].bar(x - width/2, drone_f1, width, label='Drone', color='#e74c3c', alpha=0.8, edgecolor='black')
    axes[1].bar(x + width/2, ambient_f1, width, label='Ambient', color='#f39c12', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Model', fontweight='bold')
    axes[1].set_ylabel('F1-Score', fontweight='bold')
    axes[1].set_title('F1-Score by Class', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 1.05)
    
    plt.tight_layout()
    output_path = output_dir / 'best_class_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_subcategory_heatmap(results, output_dir):
    """Heatmap de performance par sous-catégorie - test set."""
    test_results = {k: v for k, v in results.items() if k[1] == 'test'}
    
    if not test_results:
        return
    
    # Collecter toutes les sous-catégories
    all_subcats = set()
    for data in test_results.values():
        all_subcats.update(data.get('subcategory_metrics', {}).keys())
    
    if not all_subcats:
        print("  [SKIP] No subcategory metrics found")
        return
    
    # ORDRE LOGIQUE: distances décroissantes (tri numérique), puis orig_drone, puis ambients, puis orig_none
    # Séparer les catégories
    distances = []
    orig_drone = []
    ambients = []
    orig_none = []
    
    for s in all_subcats:
        if 'orig_drone' in s:
            orig_drone.append(s)
        elif 'orig_none' in s:
            orig_none.append(s)
        elif 'ambient' in s:
            ambients.append(s)
        else:
            # Extraire la distance (ex: "850m" de "850m" ou de "drone_850m")
            distances.append(s)
    
    # Trier les distances par valeur numérique DÉCROISSANTE (1000m avant 850m avant 100m)
    import re
    def extract_distance(s):
        match = re.search(r'(\d+)m', s)
        return int(match.group(1)) if match else 0
    
    distances_sorted = sorted(distances, key=extract_distance, reverse=True)
    
    # Ordre final: distances (loin → près), orig_drone, ambients, orig_none
    all_subcats = distances_sorted + sorted(orig_drone) + sorted(ambients) + sorted(orig_none)
    
    models = sorted([k[0] for k in test_results.keys()])
    
    # Créer la matrice de données
    accuracy_matrix = np.zeros((len(models), len(all_subcats)))
    
    for i, model in enumerate(models):
        key = [k for k in test_results.keys() if k[0] == model][0]
        data = test_results[key]
        subcat_metrics = data.get('subcategory_metrics', {})
        
        for j, subcat in enumerate(all_subcats):
            if subcat in subcat_metrics:
                accuracy_matrix[i, j] = subcat_metrics[subcat].get('accuracy', 0)
            else:
                accuracy_matrix[i, j] = np.nan
    
    # Plot
    fig, ax = plt.subplots(figsize=(max(14, len(all_subcats) * 0.8), max(8, len(models) * 0.6)))
    
    sns.heatmap(accuracy_matrix, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
               xticklabels=all_subcats, yticklabels=models, cbar_kws={'label': 'Accuracy'},
               vmin=0, vmax=1, linewidths=0.5, linecolor='gray')
    
    ax.set_title('Accuracy Heatmap by Subcategory (Test Set - Best Thresholds)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Subcategory', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = output_dir / 'best_subcategory_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def main():
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON - BEST RESULTS ONLY")
    print("=" * 80)
    
    print("\n[1/5] Loading best results...")
    results = load_best_results()
    
    if not results:
        print("[ERROR] No results loaded. Run select_best_results.py first!")
        return
    
    print(f"[OK] Loaded {len(results)} best configurations")
    
    print("\n[2/5] Plotting global metrics comparison...")
    plot_global_metrics_comparison(results, OUTPUT_DIR)
    
    print("\n[3/5] Plotting confusion matrices (test set)...")
    plot_confusion_matrices(results, OUTPUT_DIR)
    
    print("\n[4/5] Plotting class performance...")
    plot_class_performance(results, OUTPUT_DIR)
    
    print("\n[5/5] Plotting subcategory heatmap...")
    plot_subcategory_heatmap(results, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
