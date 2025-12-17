"""
Dataset Analysis and Visualization
Analyzes the composition and characteristics of the acoustic UAV dataset.

Features:
- Class distribution comparison (original, augmented, combined)
- Audio duration histograms
- SNR distribution for augmented samples
- Waveform and spectrogram examples
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import librosa
import librosa.display
from collections import Counter

# Import project config + plot utils
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from tools import plot_utils

# Apply consistent plotting style
plot_utils.set_style()


def count_files_in_dataset(dataset_path):
    """Count files per class in a dataset."""
    counts = {}
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        return counts
    
    for label_dir in ['0', '1']:
        label_path = dataset_path / label_dir
        if label_path.exists():
            counts[int(label_dir)] = len(list(label_path.glob('*.wav')))
        else:
            counts[int(label_dir)] = 0
    
    return counts


def plot_dataset_distribution():
    """Plot class distribution across different datasets."""
    base_path = config.EXTRACTION_DIR

    datasets = {
        'Original': config.DATASET_TEST_DIR,
        'Augmented': config.DATASET_AUGMENTED_DIR,
        'Combined': config.DATASET_COMBINED_DIR
    }
    
    data = []
    for name, path in datasets.items():
        counts = count_files_in_dataset(path)
        if counts:
            data.append({
                'Dataset': name,
                'No Drone (0)': counts.get(0, 0),
                'Drone (1)': counts.get(1, 0)
            })
    
    if not data:
        print("No dataset found!")
        return None
    
    # Create grouped bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Absolute counts
    x = np.arange(len(data))
    width = 0.35
    
    no_drone = [d['No Drone (0)'] for d in data]
    drone = [d['Drone (1)'] for d in data]
    
    ax1.bar(x - width/2, no_drone, width, label='No Drone (0)', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, drone, width, label='Drone (1)', color='coral', alpha=0.8)
    
    ax1.set_xlabel('Dataset', fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontweight='bold')
    ax1.set_title('Class Distribution - Absolute Counts', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d['Dataset'] for d in data])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (nd, dr) in enumerate(zip(no_drone, drone)):
        ax1.text(i - width/2, nd + 10, str(nd), ha='center', va='bottom', fontweight='bold')
        ax1.text(i + width/2, dr + 10, str(dr), ha='center', va='bottom', fontweight='bold')
    
    # Percentage stacked bar
    totals = [d['No Drone (0)'] + d['Drone (1)'] for d in data]
    no_drone_pct = [d['No Drone (0)'] / totals[i] * 100 if totals[i] > 0 else 0 for i, d in enumerate(data)]
    drone_pct = [d['Drone (1)'] / totals[i] * 100 if totals[i] > 0 else 0 for i, d in enumerate(data)]
    
    ax2.bar(x, no_drone_pct, width*2, label='No Drone (0)', color='steelblue', alpha=0.8)
    ax2.bar(x, drone_pct, width*2, bottom=no_drone_pct, label='Drone (1)', color='coral', alpha=0.8)
    
    ax2.set_xlabel('Dataset', fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontweight='bold')
    ax2.set_title('Class Distribution - Percentages', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([d['Dataset'] for d in data])
    ax2.legend()
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, (nd_p, dr_p) in enumerate(zip(no_drone_pct, drone_pct)):
        ax2.text(i, nd_p/2, f'{nd_p:.1f}%', ha='center', va='center', fontweight='bold', color='white')
        ax2.text(i, nd_p + dr_p/2, f'{dr_p:.1f}%', ha='center', va='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    
    # Save via plot utils
    plot_utils.save_figure(fig, 'dataset_distribution.png', script_path=__file__)
    
    return fig


def plot_snr_distribution():
    """Plot SNR distribution from augmentation metadata."""
    metadata_path = config.DATASET_AUGMENTED_DIR / "augmentation_metadata.json"
    
    if not metadata_path.exists():
        print("[WARNING] Augmentation metadata not found")
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract SNR values by category (only for drone samples)
    samples = metadata.get('samples', [])
    if not samples:
        print("[WARNING] No samples in metadata")
        return None
    
    categories = {}
    for sample in samples:
        # Skip samples without SNR (no_drone samples don't have SNR)
        if 'actual_snr_db' not in sample:
            continue
        cat = sample['category']
        snr = sample['actual_snr_db']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(snr)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot
    cat_names = list(categories.keys())
    cat_data = [categories[cat] for cat in cat_names]
    
    bp = ax1.boxplot(cat_data, labels=cat_names, patch_artist=True, showmeans=True)
    
    # Color boxes
    colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Augmentation Category', fontweight='bold')
    ax1.set_ylabel('SNR (dB)', fontweight='bold')
    ax1.set_title('SNR Distribution by Category', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Histogram
    all_snr = [snr for cat_snr in cat_data for snr in cat_snr]
    ax2.hist(all_snr, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(all_snr), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_snr):.2f} dB')
    ax2.axvline(np.median(all_snr), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_snr):.2f} dB')
    
    ax2.set_xlabel('SNR (dB)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Overall SNR Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    plot_utils.save_figure(fig, 'snr_distribution.png', script_path=__file__)
    
    return fig


def plot_audio_examples():
    """Plot waveform and spectrogram examples from each dataset."""
    dataset_combined = config.DATASET_COMBINED_DIR
    
    if not dataset_combined.exists():
        print("[WARNING] Combined dataset not found")
        return None
    
    # Get example files
    examples = {
        'No Drone (Original)': list((dataset_combined / '0').glob('orig_*.wav')),
        'Drone (Original)': list((dataset_combined / '1').glob('orig_*.wav')),
        'Drone (Very Far)': list((dataset_combined / '1').glob('drone_very_far*.wav')),
        'Drone (Close)': list((dataset_combined / '1').glob('drone_close*.wav'))
    }
    
    # Filter to get first example of each
    examples = {k: v[0] if v else None for k, v in examples.items()}
    examples = {k: v for k, v in examples.items() if v is not None}
    
    if not examples:
        print("[WARNING] No example files found")
        return None
    
    # Create figure
    n_examples = len(examples)
    fig, axes = plt.subplots(n_examples, 2, figsize=(15, 4*n_examples))
    
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (label, filepath) in enumerate(examples.items()):
        # Load audio using centralized sample rate and a short preview duration
        preview_dur = min(3.0, float(getattr(config, 'AUDIO_DURATION_S', 3.0)))
        y, sr = librosa.load(filepath, sr=config.SAMPLE_RATE, duration=preview_dur)
        
        # Waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[idx, 0], color='steelblue')
        axes[idx, 0].set_title(f'{label} - Waveform', fontweight='bold')
        axes[idx, 0].set_xlabel('Time (s)')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].grid(alpha=0.3)
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[idx, 1], cmap='viridis')
        axes[idx, 1].set_title(f'{label} - Spectrogram', fontweight='bold')
        axes[idx, 1].set_xlabel('Time (s)')
        axes[idx, 1].set_ylabel('Frequency (Hz)')
        fig.colorbar(img, ax=axes[idx, 1], format='%+2.0f dB')
    
    plt.tight_layout()
    
    plot_utils.save_figure(fig, 'audio_examples.png', script_path=__file__)
    
    return fig


def generate_summary_stats():
    """Generate and save summary statistics."""
    base_path = config.EXTRACTION_DIR
    
    stats = {
        'datasets': {},
        'augmentation': {}
    }
    
    # Dataset counts
    for name in ['dataset_test', 'dataset_augmented', 'dataset_combined']:
        dataset_path = base_path / name
        counts = count_files_in_dataset(dataset_path)
        if counts:
            total = sum(counts.values())
            stats['datasets'][name] = {
                'no_drone': counts.get(0, 0),
                'drone': counts.get(1, 0),
                'total': total,
                'balance_ratio': counts.get(1, 0) / counts.get(0, 1) if counts.get(0, 0) > 0 else 0
            }
    
    # Augmentation metadata
    metadata_path = base_path / "dataset_augmented" / "augmentation_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            aug_data = json.load(f)
        
        stats['augmentation'] = aug_data.get('statistics', {})
    
    # Save stats
    stats_output_dir = plot_utils.get_output_dir(__file__)
    stats_path = stats_output_dir / 'dataset_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"[OK] Saved: {stats_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("DATASET SUMMARY STATISTICS")
    print("="*80)
    
    for dataset_name, dataset_stats in stats['datasets'].items():
        print(f"\n{dataset_name}:")
        print(f"  No Drone: {dataset_stats['no_drone']}")
        print(f"  Drone: {dataset_stats['drone']}")
        print(f"  Total: {dataset_stats['total']}")
        print(f"  Balance Ratio (Drone/No-Drone): {dataset_stats['balance_ratio']:.2f}")
    
    print("\n" + "="*80)
    
    return stats


def main():
    """Run all dataset analysis visualizations."""
    print("="*80)
    print("DATASET ANALYSIS")
    print("="*80 + "\n")
    
    # Generate summary statistics
    print("[1/4] Generating summary statistics...")
    generate_summary_stats()
    
    # Plot dataset distribution
    print("\n[2/4] Plotting dataset distribution...")
    plot_dataset_distribution()
    
    # Plot SNR distribution
    print("\n[3/4] Plotting SNR distribution...")
    plot_snr_distribution()
    
    # Plot audio examples
    print("\n[4/4] Plotting audio examples...")
    plot_audio_examples()
    
    print("\n" + "="*80)
    print("[SUCCESS] Dataset analysis complete!")
    print("="*80)
    print(f"\nOutputs saved in: {Path(__file__).parent / 'outputs'}")


if __name__ == "__main__":
    main()
