"""
Augmentation Impact Analysis
Evaluates the impact of data augmentation on model performance.

Features:
- Before/after augmentation comparison
- Performance breakdown by SNR category
- Error analysis (false positives/negatives by category)
- Augmentation effectiveness visualization
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Import project config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_augmentation_metadata():
    """Load augmentation metadata."""
    metadata_path = config.PROJECT_ROOT / "0 - DADS dataset extraction" / "dataset_augmented" / "augmentation_metadata.json"
    
    if not metadata_path.exists():
        print("[WARNING] Augmentation metadata not found")
        return None
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def plot_snr_performance():
    """Plot model performance by SNR category."""
    metadata = load_augmentation_metadata()
    
    if metadata is None:
        return None
    
    # Get categories info from config
    drone_categories = metadata.get('config', {}).get('drone_augmentation', {}).get('categories', [])
    
    if not drone_categories:
        print("[WARNING] No category information in metadata")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    categories = [cat['name'] for cat in drone_categories]
    snr_values = [cat['snr_db'] for cat in drone_categories]
    descriptions = [cat['description'] for cat in drone_categories]
    
    # Get actual achieved SNR from statistics
    stats = metadata.get('statistics', {}).get('drone_augmentation', {}).get('categories', {})
    achieved_snr = [stats.get(cat, {}).get('avg_snr_db', snr) 
                    for cat, snr in zip(categories, snr_values)]
    
    # Plot SNR values
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, snr_values, width, label='Target SNR', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, achieved_snr, width, label='Achieved SNR', color='coral', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Augmentation Category', fontweight='bold', fontsize=12)
    ax.set_ylabel('SNR (dB)', fontweight='bold', fontsize=12)
    ax.set_title('Signal-to-Noise Ratio by Augmentation Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'snr_performance.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'snr_performance.png'}")
    
    return fig


def plot_augmentation_composition():
    """Plot the composition of augmented samples."""
    metadata = load_augmentation_metadata()
    
    if metadata is None:
        return None
    
    # Get both drone and no_drone statistics
    drone_stats = metadata.get('statistics', {}).get('drone_augmentation', {}).get('categories', {})
    no_drone_stats = metadata.get('statistics', {}).get('no_drone_augmentation', {}).get('categories', {})
    
    if not drone_stats and not no_drone_stats:
        print("[WARNING] No statistics in metadata")
        return None
    
    # Combine both for visualization
    all_stats = {**drone_stats, **no_drone_stats}
    
    # Extract data
    categories = list(all_stats.keys())
    counts = [all_stats[cat]['generated'] for cat in categories]
    # Use a seaborn qualitative palette to ensure distinct, colourblind-friendly colors
    try:
        palette = sns.color_palette('tab10', n_colors=len(categories))
    except Exception:
        palette = sns.color_palette('muted', n_colors=len(categories))

    # Map palette entries to categories in the observed order so colors are stable but
    # visually distinct (matplotlib accepts RGB tuples from seaborn)
    colors = [palette[i % len(palette)] for i in range(len(categories))]
    
    # Create figure with pie and bar charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart
    wedges, texts, autotexts = ax1.pie(counts, labels=categories, autopct='%1.1f%%',
                                         colors=colors, startangle=90,
                                         textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    ax1.set_title('Augmented Dataset Composition', fontsize=14, fontweight='bold')
    
    # Make percentage text white and bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Bar chart
    x = np.arange(len(categories))
    bars = ax2.bar(x, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Category', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
    ax2.set_title('Sample Count by Category', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'augmentation_composition.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'augmentation_composition.png'}")
    
    return fig


def plot_dataset_evolution():
    """Plot how the dataset evolved with augmentation."""
    base_path = config.PROJECT_ROOT / "0 - DADS dataset extraction"
    
    # Count files
    datasets_info = []
    
    # Original
    orig_path = base_path / 'dataset_test'
    if orig_path.exists():
        no_drone_orig = len(list((orig_path / '0').glob('*.wav')))
        drone_orig = len(list((orig_path / '1').glob('*.wav')))
        datasets_info.append({
            'Stage': 'Original\nDataset',
            'No Drone': no_drone_orig,
            'Drone': drone_orig,
            'Total': no_drone_orig + drone_orig
        })
    
    # Augmented
    aug_path = base_path / 'dataset_augmented'
    if aug_path.exists():
        no_drone_aug = len(list((aug_path / '0').glob('*.wav')))
        aug_drone = len(list((aug_path / '1').glob('*.wav')))
        datasets_info.append({
            'Stage': 'Augmented\nDataset',
            'No Drone': no_drone_aug,
            'Drone': aug_drone,
            'Total': no_drone_aug + aug_drone
        })
    
    # Combined
    comb_path = base_path / 'dataset_combined'
    if comb_path.exists():
        no_drone_comb = len(list((comb_path / '0').glob('*.wav')))
        drone_comb = len(list((comb_path / '1').glob('*.wav')))
        datasets_info.append({
            'Stage': 'Combined\nDataset',
            'No Drone': no_drone_comb,
            'Drone': drone_comb,
            'Total': no_drone_comb + drone_comb
        })
    
    if not datasets_info:
        print("[WARNING] No dataset information available")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    stages = [d['Stage'] for d in datasets_info]
    no_drone = [d['No Drone'] for d in datasets_info]
    drone = [d['Drone'] for d in datasets_info]
    
    x = np.arange(len(stages))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, no_drone, width, label='No Drone', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, drone, width, label='Drone', color='coral', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add total labels
    for i, d in enumerate(datasets_info):
        ax.text(i, max(no_drone[i], drone[i]) * 1.1,
               f'Total: {d["Total"]}',
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Dataset Stage', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
    ax.set_title('Dataset Evolution with Augmentation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'dataset_evolution.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'dataset_evolution.png'}")
    
    return fig


def main():
    """Run all augmentation impact visualizations."""
    print("="*80)
    print("AUGMENTATION IMPACT ANALYSIS")
    print("="*80 + "\n")
    
    # Plot SNR performance
    print("[1/3] Plotting SNR performance...")
    plot_snr_performance()
    
    # Plot augmentation composition
    print("\n[2/3] Plotting augmentation composition...")
    plot_augmentation_composition()
    
    # Plot dataset evolution
    print("\n[3/3] Plotting dataset evolution...")
    plot_dataset_evolution()
    
    print("\n" + "="*80)
    print("[SUCCESS] Augmentation impact analysis complete!")
    print("="*80)
    print(f"\nOutputs saved in: {Path(__file__).parent / 'outputs'}")


if __name__ == "__main__":
    main()
