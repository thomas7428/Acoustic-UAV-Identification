"""
Modern Dataset Analysis
Analyse la composition et les statistiques du dataset à partir de config.py.

Features:
- Utilise config.DATASET_TEST_DIR, DATASET_TRAIN_DIR, DATASET_VAL_DIR
- Statistiques par split (train/val/test)
- Distribution par catégorie (distances, ambient types, originals)
- Visualisations de la composition du dataset
- Pas de fallback, configuration centralisée
"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
import re

# Import project config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from tools import plot_utils

# Apply plotting style
plot_utils.set_style()

# Output directory
OUTPUT_DIR = plot_utils.get_output_dir(__file__)


def analyze_split(split_dir, split_name):
    """
    Analyse un split (train/val/test) et retourne les statistiques.
    """
    stats = {
        'split': split_name,
        'total': 0,
        'class_0': 0,
        'class_1': 0,
        'subcategories': defaultdict(int),
        'files_by_class': {'0': [], '1': []}
    }
    
    if not split_dir.exists():
        print(f"  ⚠ Split directory not found: {split_dir}")
        return stats
    
    # Analyser class 0 et class 1
    for class_label in ['0', '1']:
        class_dir = split_dir / class_label
        if not class_dir.exists():
            continue
        
        wav_files = list(class_dir.glob('*.wav'))
        stats['files_by_class'][class_label] = wav_files
        
        if class_label == '0':
            stats['class_0'] = len(wav_files)
        else:
            stats['class_1'] = len(wav_files)
        
        stats['total'] += len(wav_files)
        
        # Analyser les sous-catégories
        for wav_file in wav_files:
            fname = wav_file.name.lower()
            
            # Catégoriser
            if 'dads_0_' in fname:
                stats['subcategories']['orig_ambient'] += 1
            elif 'dads_1_' in fname:
                stats['subcategories']['orig_drone'] += 1
            elif 'ambient' in fname:
                # Extraire type d'ambient
                if 'complex' in fname:
                    stats['subcategories']['ambient_complex'] += 1
                elif 'extreme' in fname:
                    stats['subcategories']['ambient_extreme'] += 1
                elif 'moderate' in fname:
                    stats['subcategories']['ambient_moderate'] += 1
                elif 'very' in fname:
                    stats['subcategories']['ambient_very'] += 1
                else:
                    stats['subcategories']['ambient_other'] += 1
            else:
                # Extraire distance pour les drones
                match = re.search(r'(\d+)m', fname)
                if match:
                    distance = match.group(1)
                    stats['subcategories'][f'drone_{distance}m'] += 1
                else:
                    stats['subcategories']['other'] += 1
    
    return stats


def plot_split_distribution(all_stats, output_dir):
    """
    Plot la distribution des classes par split.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    splits = [s['split'] for s in all_stats]
    class_0_counts = [s['class_0'] for s in all_stats]
    class_1_counts = [s['class_1'] for s in all_stats]
    
    x = np.arange(len(splits))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, class_0_counts, width, label='Class 0 (Non-Drone)', 
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, class_1_counts, width, label='Class 1 (Drone)',
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Split', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
    ax.set_title('Dataset Distribution by Split', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Ajouter valeurs sur les barres
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / 'dataset_split_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_subcategory_distribution(all_stats, output_dir):
    """
    Plot la distribution des sous-catégories pour le split de test.
    """
    # Utiliser les stats du test split
    test_stats = next((s for s in all_stats if s['split'] == 'test'), None)
    if not test_stats or not test_stats['subcategories']:
        print("  ⚠ No test split subcategories to plot")
        return
    
    # Séparer drones et ambients
    drone_cats = {}
    ambient_cats = {}
    
    for cat, count in test_stats['subcategories'].items():
        if 'drone' in cat:
            drone_cats[cat] = count
        else:
            ambient_cats[cat] = count
    
    # Plot drones
    if drone_cats:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Trier par distance
        def extract_dist(cat_name):
            if 'orig' in cat_name:
                return 0
            match = re.search(r'(\d+)', cat_name)
            return int(match.group(1)) if match else 999
        
        sorted_cats = sorted(drone_cats.items(), key=lambda x: extract_dist(x[0]))
        labels = [cat.replace('_', ' ').title() for cat, _ in sorted_cats]
        values = [count for _, count in sorted_cats]
        
        bars = ax.bar(range(len(labels)), values, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Drone Category', fontweight='bold', fontsize=12)
        ax.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
        ax.set_title('Drone Distribution by Distance (Test Set)', fontweight='bold', fontsize=14)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = output_dir / 'dataset_drone_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()
    
    # Plot ambients
    if ambient_cats:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sorted_cats = sorted(ambient_cats.items())
        labels = [cat.replace('_', ' ').title() for cat, _ in sorted_cats]
        values = [count for _, count in sorted_cats]
        
        bars = ax.bar(range(len(labels)), values, color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Ambient Category', fontweight='bold', fontsize=12)
        ax.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
        ax.set_title('Ambient Distribution by Type (Test Set)', fontweight='bold', fontsize=14)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = output_dir / 'dataset_ambient_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()


def generate_summary_report(all_stats, output_dir):
    """
    Génère un rapport texte avec toutes les statistiques.
    """
    report_path = output_dir / 'dataset_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DATASET ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Stats globales
        total_samples = sum(s['total'] for s in all_stats)
        total_class_0 = sum(s['class_0'] for s in all_stats)
        total_class_1 = sum(s['class_1'] for s in all_stats)
        
        f.write(f"GLOBAL STATISTICS:\n")
        f.write(f"  Total Samples: {total_samples}\n")
        f.write(f"  Class 0 (Non-Drone): {total_class_0} ({total_class_0/total_samples*100:.1f}%)\n")
        f.write(f"  Class 1 (Drone): {total_class_1} ({total_class_1/total_samples*100:.1f}%)\n")
        f.write(f"  Class Balance: {min(total_class_0, total_class_1) / max(total_class_0, total_class_1):.2f}\n\n")
        
        # Stats par split
        for stats in all_stats:
            f.write(f"\n{'='*80}\n")
            f.write(f"{stats['split'].upper()} SPLIT\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"  Total: {stats['total']} samples\n")
            f.write(f"  Class 0: {stats['class_0']} ({stats['class_0']/stats['total']*100:.1f}%)\n")
            f.write(f"  Class 1: {stats['class_1']} ({stats['class_1']/stats['total']*100:.1f}%)\n\n")
            
            if stats['subcategories']:
                f.write(f"  Subcategories:\n")
                for cat, count in sorted(stats['subcategories'].items(), key=lambda x: -x[1]):
                    f.write(f"    {cat:25s}: {count:4d} samples\n")
    
    print(f"  ✓ Saved: {report_path}")


def main():
    """Point d'entrée principal."""
    print("\n" + "="*80)
    print("DATASET ANALYSIS")
    print("="*80 + "\n")
    
    # Analyser tous les splits
    print("[1/4] Analyzing splits...")
    all_stats = []
    
    splits = [
        ('train', config.DATASET_TRAIN_DIR),
        ('val', config.DATASET_VAL_DIR),
        ('test', config.DATASET_TEST_DIR)
    ]
    
    for split_name, split_dir in splits:
        print(f"  Analyzing {split_name}...")
        stats = analyze_split(split_dir, split_name)
        all_stats.append(stats)
        print(f"    Total: {stats['total']}, Class 0: {stats['class_0']}, Class 1: {stats['class_1']}")
    
    # Générer les visualisations
    print("\n[2/4] Generating split distribution plot...")
    plot_split_distribution(all_stats, OUTPUT_DIR)
    
    print("\n[3/4] Generating subcategory distribution plots...")
    plot_subcategory_distribution(all_stats, OUTPUT_DIR)
    
    print("\n[4/4] Generating summary report...")
    generate_summary_report(all_stats, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print(f"✓ Dataset analysis complete! Results in: {OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
