"""
Performance Comparison Visualization
Visualisation complète des résultats de performance des modèles à partir des JSON précalculés.

Features:
- Chargement automatique des résultats depuis config.PERFORMANCE_DIR
- Comparaison multi-modèles et multi-splits
- Union de résultats (train + test, différents thresholds)
- Métriques globales, par classe, et par sous-catégorie
- Matrices de confusion, courbes de performance, analyse par distance/ambient

Usage:
    python performance_comparison.py --models CNN RNN CRNN --splits test val
    python performance_comparison.py --threshold 0.5 0.6  # Compare différents thresholds
    python performance_comparison.py --all  # Tous les fichiers disponibles
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import argparse
from datetime import datetime

# Import project config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from tools import plot_utils

# Apply consistent plotting style
plot_utils.set_style()

# Output directory
OUTPUT_DIR = plot_utils.get_output_dir(__file__)


def load_performance_results(models=None, splits=None, thresholds=None):
    """
    Charge les fichiers JSON de performance depuis config.PERFORMANCE_DIR.
    
    Args:
        models: Liste des modèles (ex: ['CNN', 'RNN']) ou None pour tous
        splits: Liste des splits (ex: ['test', 'val']) ou None pour tous
        thresholds: Liste des thresholds (ex: [0.5, 0.6]) ou None pour tous
    
    Returns:
        Dict de résultats groupés par (model, split, threshold)
    """
    perf_dir = config.PERFORMANCE_DIR
    
    if not perf_dir.exists():
        print(f"[ERROR] Performance directory not found: {perf_dir}")
        return {}
    
    # Pattern: {model}_{split}_t{threshold:.2f}_{timestamp}.json
    # Now expect canonical filenames without timestamps. Load files and use metadata.
    json_files = list(perf_dir.glob("*.json"))

    if not json_files:
        print(f"[ERROR] No JSON files found in {perf_dir}")
        return {}

    print(f"[INFO] Found {len(json_files)} performance files")

    results = {}

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            metadata = data.get('metadata', {})
            model = metadata.get('model', 'UNKNOWN')
            split = metadata.get('split', 'UNKNOWN')
            threshold = metadata.get('threshold', 0.5)

            # Filtres
            if models and model not in models:
                continue
            if splits and split not in splits:
                continue
            if thresholds and threshold not in thresholds:
                continue

            key = (model, split, threshold)

            # If duplicate keys, overwrite and warn (we assume canonical filenames)
            if key in results:
                print(f"[WARN] Duplicate result for {key}, overwriting with {json_file.name}")

            results[key] = data
            print(f"  ✓ Loaded: {model} | {split} | t={threshold:.2f} | {json_file.name}")

        except Exception as e:
            print(f"  ✗ Error loading {json_file.name}: {e}")

    print(f"[OK] Loaded {len(results)} result sets")
    return results


def plot_global_metrics_comparison(results, output_dir):
    """
    Compare les métriques globales entre tous les modèles/splits/thresholds.
    Génère un bar chart avec toutes les métriques clés.
    """
    if not results:
        print("[SKIP] No results to plot global metrics")
        return
    
    # Extraire les données
    rows = []
    for (model, split, threshold), data in results.items():
        metrics = data.get('global_metrics', {})
        rows.append({
            'model': model,
            'split': split,
            'threshold': threshold,
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0),
            'specificity': metrics.get('specificity', 0),
        })
    
    # Créer figure avec subplots pour chaque métrique
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Grouper par model et threshold pour le plot
        x_labels = []
        values = []
        colors = []
        
        color_map = {'CNN': '#e74c3c', 'RNN': '#3498db', 'CRNN': '#2ecc71', 'ATTENTION_CRNN': '#f39c12'}
        
        for row in rows:
            label = f"{row['model']}\n{row['split']}\nt={row['threshold']:.2f}"
            x_labels.append(label)
            values.append(row[metric])
            colors.append(color_map.get(row['model'], '#95a5a6'))
        
        bars = ax.bar(range(len(values)), values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Ajouter valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=7)
    
    # Supprimer subplot vide
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    output_path = output_dir / 'global_metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_confusion_matrices(results, output_dir):
    """
    Affiche les matrices de confusion pour chaque résultat.
    """
    if not results:
        print("[SKIP] No results to plot confusion matrices")
        return
    
    n_results = len(results)
    cols = min(3, n_results)
    rows = (n_results + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_results == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_results > 1 else [axes]
    
    for idx, ((model, split, threshold), data) in enumerate(results.items()):
        ax = axes[idx]
        
        metrics = data.get('global_metrics', {})
        tp = metrics.get('tp', 0)
        tn = metrics.get('tn', 0)
        fp = metrics.get('fp', 0)
        fn = metrics.get('fn', 0)
        
        # Matrice de confusion: [[TN, FP], [FN, TP]]
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Plot avec seaborn
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Non-Drone', 'Drone'],
                   yticklabels=['Non-Drone', 'Drone'],
                   cbar=False)
        
        ax.set_title(f'{model} | {split} | t={threshold:.2f}\nAcc={metrics.get("accuracy", 0):.3f}',
                    fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    # Supprimer axes vides
    for idx in range(n_results, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    output_path = output_dir / 'confusion_matrices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_class_performance(results, output_dir):
    """
    Compare les performances par classe (class 0 vs class 1).
    """
    if not results:
        print("[SKIP] No results to plot class performance")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Données pour class 0 et class 1
    x_labels = []
    class0_acc = []
    class1_acc = []
    class1_prec = []
    class1_rec = []
    class1_f1 = []
    colors = []
    
    color_map = {'CNN': '#e74c3c', 'RNN': '#3498db', 'CRNN': '#2ecc71', 'ATTENTION_CRNN': '#f39c12'}
    
    for (model, split, threshold), data in results.items():
        class_metrics = data.get('class_metrics', {})
        
        c0 = class_metrics.get('class_0', {})
        c1 = class_metrics.get('class_1', {})
        
        label = f"{model}\n{split}\nt={threshold:.2f}"
        x_labels.append(label)
        class0_acc.append(c0.get('accuracy', 0))
        class1_acc.append(c1.get('accuracy', 0))
        class1_prec.append(c1.get('precision', 0))
        class1_rec.append(c1.get('recall', 0))
        class1_f1.append(c1.get('f1_score', 0))
        colors.append(color_map.get(model, '#95a5a6'))
    
    x = np.arange(len(x_labels))
    width = 0.35
    
    # Subplot 1: Accuracy par classe
    bars1 = ax1.bar(x - width/2, class0_acc, width, label='Class 0 (Non-Drone)', 
                    color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x + width/2, class1_acc, width, label='Class 1 (Drone)',
                    color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy by Class', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Ajouter valeurs
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=7)
    
    # Subplot 2: Métriques class 1
    width2 = 0.2
    bars1 = ax2.bar(x - width2*1.5, class1_acc, width2, label='Accuracy (Recall)', 
                    color='#2ecc71', alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x - width2*0.5, class1_prec, width2, label='Precision',
                    color='#3498db', alpha=0.7, edgecolor='black')
    bars3 = ax2.bar(x + width2*0.5, class1_rec, width2, label='Recall',
                    color='#e74c3c', alpha=0.7, edgecolor='black')
    bars4 = ax2.bar(x + width2*1.5, class1_f1, width2, label='F1-Score',
                    color='#f39c12', alpha=0.7, edgecolor='black')
    
    ax2.set_ylabel('Score')
    ax2.set_title('Class 1 (Drone) Detailed Metrics', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'class_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_subcategory_performance(results, output_dir):
    """
    Analyse de performance par sous-catégories (distances drones, types ambient).
    Génère des plots séparés pour drones et non-drones.
    """
    if not results:
        print("[SKIP] No results to plot subcategory performance")
        return
    
    # Extraire toutes les sous-catégories
    all_drone_cats = set()
    all_ambient_cats = set()
    
    for data in results.values():
        subcat_metrics = data.get('subcategory_metrics', {})
        for cat_name, cat_data in subcat_metrics.items():
            label = cat_data.get('label', -1)
            if label == 1 or 'drone' in cat_name.lower():
                all_drone_cats.add(cat_name)
            else:
                all_ambient_cats.add(cat_name)
    
    # Plot 1: Drones par distance
    if all_drone_cats:
        plot_drone_distances(results, all_drone_cats, output_dir)
    
    # Plot 2: Ambients par type
    if all_ambient_cats:
        plot_ambient_types(results, all_ambient_cats, output_dir)


def plot_drone_distances(results, drone_cats, output_dir):
    """Plot performance des drones par distance."""
    # Trier les catégories par distance
    def extract_distance(cat_name):
        if 'orig' in cat_name.lower():
            return 0
        # Extraire nombre (ex: "100m" -> 100)
        import re
        match = re.search(r'(\d+)m?', cat_name)
        return int(match.group(1)) if match else 999
    
    sorted_cats = sorted(drone_cats, key=extract_distance)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Pour chaque modèle, tracer une courbe
    for (model, split, threshold), data in results.items():
        subcat_metrics = data.get('subcategory_metrics', {})
        
        distances = []
        accuracies = []
        f1_scores = []
        
        for cat in sorted_cats:
            if cat not in subcat_metrics:
                continue
            cat_data = subcat_metrics[cat]
            dist = extract_distance(cat)
            acc = cat_data.get('accuracy', 0)
            f1 = cat_data.get('f1_score', 0)
            
            distances.append(dist)
            accuracies.append(acc)
            f1_scores.append(f1)
        
        if not distances:
            continue
        
        label = f"{model} {split} t={threshold:.2f}"
        ax1.plot(distances, accuracies, marker='o', label=label, linewidth=2, markersize=6)
        ax2.plot(distances, f1_scores, marker='s', label=label, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Distance (m)', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Drone Detection Accuracy by Distance', fontweight='bold', fontsize=14)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    ax2.set_xlabel('Distance (m)', fontweight='bold')
    ax2.set_ylabel('F1-Score', fontweight='bold')
    ax2.set_title('Drone Detection F1-Score by Distance', fontweight='bold', fontsize=14)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    output_path = output_dir / 'drone_performance_by_distance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_ambient_types(results, ambient_cats, output_dir):
    """Plot performance des ambients par type."""
    sorted_cats = sorted(ambient_cats)
    
    n_results = len(results)
    fig, ax = plt.subplots(figsize=(max(12, len(sorted_cats)*2), 8))
    
    x = np.arange(len(sorted_cats))
    width = 0.8 / n_results
    
    for idx, ((model, split, threshold), data) in enumerate(results.items()):
        subcat_metrics = data.get('subcategory_metrics', {})
        
        accuracies = []
        for cat in sorted_cats:
            if cat in subcat_metrics:
                accuracies.append(subcat_metrics[cat].get('accuracy', 0))
            else:
                accuracies.append(0)
        
        offset = (idx - n_results/2) * width + width/2
        label = f"{model} {split} t={threshold:.2f}"
        bars = ax.bar(x + offset, accuracies, width, label=label, alpha=0.7, edgecolor='black')
        
        # Ajouter valeurs sur barres
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=7, rotation=0)
    
    ax.set_xlabel('Ambient Type', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Non-Drone (Ambient) Detection by Type', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_cats, rotation=45, ha='right')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    output_path = output_dir / 'ambient_performance_by_type.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_threshold_impact(results, output_dir):
    """
    Analyse l'impact des différents thresholds sur les performances.
    Compare pour un même modèle/split différents thresholds.
    """
    # Grouper par (model, split)
    grouped = defaultdict(list)
    for (model, split, threshold), data in results.items():
        grouped[(model, split)].append((threshold, data))
    
    # Filtrer ceux qui ont plusieurs thresholds
    multi_threshold = {k: v for k, v in grouped.items() if len(v) > 1}
    
    if not multi_threshold:
        print("[SKIP] No multi-threshold results to compare")
        return
    
    for (model, split), threshold_data in multi_threshold.items():
        # Trier par threshold
        threshold_data.sort(key=lambda x: x[0])
        
        thresholds = [t for t, _ in threshold_data]
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        fps = []
        fns = []
        
        for t, data in threshold_data:
            metrics = data.get('global_metrics', {})
            accuracies.append(metrics.get('accuracy', 0))
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1_score', 0))
            fps.append(metrics.get('fp', 0))
            fns.append(metrics.get('fn', 0))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Subplot 1: Métriques principales
        ax1.plot(thresholds, accuracies, marker='o', label='Accuracy', linewidth=2, markersize=8)
        ax1.plot(thresholds, precisions, marker='s', label='Precision', linewidth=2, markersize=8)
        ax1.plot(thresholds, recalls, marker='^', label='Recall', linewidth=2, markersize=8)
        ax1.plot(thresholds, f1_scores, marker='d', label='F1-Score', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Decision Threshold', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title(f'Threshold Impact on Metrics\n{model} | {split}', fontweight='bold', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        # Subplot 2: FP vs FN trade-off
        ax2.plot(thresholds, fps, marker='o', label='False Positives', linewidth=2, markersize=8, color='#e74c3c')
        ax2.plot(thresholds, fns, marker='s', label='False Negatives', linewidth=2, markersize=8, color='#3498db')
        
        ax2.set_xlabel('Decision Threshold', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title(f'FP vs FN Trade-off\n{model} | {split}', fontweight='bold', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / f'threshold_impact_{model}_{split}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()


def generate_summary_report(results, output_dir):
    """
    Génère un rapport texte avec toutes les métriques principales.
    """
    report_path = output_dir / 'performance_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PERFORMANCE SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for (model, split, threshold), data in sorted(results.items()):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Model: {model} | Split: {split} | Threshold: {threshold:.2f}\n")
            f.write(f"{'=' * 80}\n\n")
            
            # Global metrics
            metrics = data.get('global_metrics', {})
            f.write("GLOBAL METRICS:\n")
            f.write(f"  Total Samples: {metrics.get('total', 0)}\n")
            f.write(f"  Accuracy:      {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"  Precision:     {metrics.get('precision', 0):.4f}\n")
            f.write(f"  Recall:        {metrics.get('recall', 0):.4f}\n")
            f.write(f"  F1-Score:      {metrics.get('f1_score', 0):.4f}\n")
            f.write(f"  Specificity:   {metrics.get('specificity', 0):.4f}\n")
            f.write(f"  TP={metrics.get('tp', 0)}, TN={metrics.get('tn', 0)}, ")
            f.write(f"FP={metrics.get('fp', 0)}, FN={metrics.get('fn', 0)}\n\n")
            
            # Class metrics
            class_metrics = data.get('class_metrics', {})
            f.write("CLASS METRICS:\n")
            for class_name, class_data in sorted(class_metrics.items()):
                f.write(f"  {class_name}:\n")
                f.write(f"    Total: {class_data.get('total', 0)}\n")
                f.write(f"    Accuracy: {class_data.get('accuracy', 0):.4f}\n")
                if 'precision' in class_data:
                    f.write(f"    Precision: {class_data.get('precision', 0):.4f}\n")
                    f.write(f"    Recall: {class_data.get('recall', 0):.4f}\n")
                    f.write(f"    F1-Score: {class_data.get('f1_score', 0):.4f}\n")
            
            f.write("\n")
            
            # Top/Bottom subcategories
            subcat_metrics = data.get('subcategory_metrics', {})
            if subcat_metrics:
                # Trier par accuracy
                sorted_subcats = sorted(subcat_metrics.items(), 
                                       key=lambda x: x[1].get('accuracy', 0),
                                       reverse=True)
                
                f.write("TOP 5 SUBCATEGORIES (by accuracy):\n")
                for cat_name, cat_data in sorted_subcats[:5]:
                    f.write(f"  {cat_name:20s}: Acc={cat_data.get('accuracy', 0):.4f}, ")
                    f.write(f"F1={cat_data.get('f1_score', 0):.4f}, ")
                    f.write(f"Total={cat_data.get('total', 0)}\n")
                
                f.write("\nBOTTOM 5 SUBCATEGORIES (by accuracy):\n")
                for cat_name, cat_data in sorted_subcats[-5:]:
                    f.write(f"  {cat_name:20s}: Acc={cat_data.get('accuracy', 0):.4f}, ")
                    f.write(f"F1={cat_data.get('f1_score', 0):.4f}, ")
                    f.write(f"Total={cat_data.get('total', 0)}\n")
    
    print(f"  ✓ Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Performance Comparison Visualization')
    parser.add_argument('--models', nargs='+', help='Models to include (CNN, RNN, CRNN, ATTENTION_CRNN)')
    parser.add_argument('--splits', nargs='+', help='Splits to include (train, val, test)')
    parser.add_argument('--thresholds', nargs='+', type=float, help='Thresholds to include (0.5, 0.6, etc.)')
    parser.add_argument('--all', action='store_true', help='Include all available results')
    
    args = parser.parse_args()
    
    # Si --all, ne pas filtrer
    models = None if args.all else args.models
    splits = None if args.all else args.splits
    thresholds = None if args.all else args.thresholds
    
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON VISUALIZATION")
    print("=" * 80)
    
    # Charger les résultats
    print("\n[1/7] Loading performance results...")
    results = load_performance_results(models, splits, thresholds)
    
    if not results:
        print("[ERROR] No results loaded. Exiting.")
        return
    
    # Générer les visualisations
    print("\n[2/7] Plotting global metrics comparison...")
    plot_global_metrics_comparison(results, OUTPUT_DIR)
    
    print("\n[3/7] Plotting confusion matrices...")
    plot_confusion_matrices(results, OUTPUT_DIR)
    
    print("\n[4/7] Plotting class performance...")
    plot_class_performance(results, OUTPUT_DIR)
    
    print("\n[5/7] Plotting subcategory performance...")
    plot_subcategory_performance(results, OUTPUT_DIR)
    
    print("\n[6/7] Analyzing threshold impact...")
    plot_threshold_impact(results, OUTPUT_DIR)
    
    print("\n[7/7] Generating summary report...")
    generate_summary_report(results, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
