"""
Dataset Train/Validation/Test Splitter
Splits a source dataset into separate train, validation, and test sets with NO overlap.

Key Features:
- Ensures zero data leakage between splits
- Maintains class balance in all splits
- Reproducible with random seed
- Saves split indices for transparency
- Creates ready-to-use directory structures

Usage:
    python split_dataset.py --source dataset_combined --train 0.7 --val 0.15 --test 0.15
    python split_dataset.py --source dataset_combined --train 0.8 --test 0.2 --no-val
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import random
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_files_by_class(source_dir):
    """
    Get all audio files organized by class.
    
    Returns:
        dict: {class_label: [list of file paths]}
    """
    source_path = Path(source_dir)
    files_by_class = {}
    
    # Look for subdirectories named '0' and '1' (class labels)
    for class_dir in ['0', '1']:
        class_path = source_path / class_dir
        if class_path.exists() and class_path.is_dir():
            wav_files = sorted(list(class_path.glob('*.wav')))
            if wav_files:
                files_by_class[class_dir] = wav_files
    
    return files_by_class


def split_files(files, train_ratio, val_ratio, test_ratio):
    """
    Split list of files into train/val/test sets.
    
    Args:
        files: List of file paths
        train_ratio: Proportion for training (e.g., 0.7)
        val_ratio: Proportion for validation (e.g., 0.15)
        test_ratio: Proportion for test (e.g., 0.15)
    
    Returns:
        tuple: (train_files, val_files, test_files, indices_dict)
    """
    # Shuffle files
    files = list(files)
    random.shuffle(files)
    
    n_total = len(files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # Test gets remainder to ensure all files are used
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    
    # Store indices for transparency
    indices = {
        'total_files': n_total,
        'train_indices': list(range(0, n_train)),
        'val_indices': list(range(n_train, n_train + n_val)),
        'test_indices': list(range(n_train + n_val, n_total)),
        'train_count': len(train_files),
        'val_count': len(val_files),
        'test_count': len(test_files)
    }
    
    return train_files, val_files, test_files, indices


def copy_files(files, dest_dir, class_label, desc="Copying"):
    """Copy files to destination directory with progress bar."""
    dest_path = Path(dest_dir) / class_label
    dest_path.mkdir(parents=True, exist_ok=True)
    
    for file_path in tqdm(files, desc=f"{desc} class {class_label}"):
        dest_file = dest_path / file_path.name
        shutil.copy2(file_path, dest_file)


def create_dataset_splits(source_dir, output_base, train_ratio, val_ratio, test_ratio, 
                          random_seed=42, copy_files_flag=True, dry_run=False):
    """
    Create train/validation/test splits from source dataset.
    
    Args:
        source_dir: Source directory containing class subdirectories (0/ and 1/)
        output_base: Base output directory (will create train/val/test subdirs)
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set (0 to skip validation)
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        copy_files_flag: If True, copy files; if False, just report split
        dry_run: If True, don't create any files
    
    Returns:
        dict: Statistics about the split
    """
    # Set random seed
    set_random_seed(random_seed)
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0 (got {total_ratio})")
    
    # Get source files by class
    print(f"\n{'='*80}")
    print("DATASET SPLITTING")
    print(f"{'='*80}")
    print(f"Source directory: {source_dir}")
    print(f"Output base: {output_base}")
    print(f"Split ratios - Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
    print(f"Random seed: {random_seed}")
    
    if dry_run:
        print("\n[!] DRY RUN MODE - No files will be created\n")
    
    files_by_class = get_files_by_class(source_dir)
    
    if not files_by_class:
        print("[ERROR] No class directories (0/ or 1/) found in source directory!")
        return None
    
    print(f"\nFound classes: {list(files_by_class.keys())}")
    for class_label, files in files_by_class.items():
        print(f"  Class {class_label}: {len(files)} files")
    
    # Create output directories (use canonical names from config)
    output_path = Path(output_base)
    train_dir = output_path / Path(config.DATASET_TRAIN_DIR).name
    val_dir = output_path / Path(config.DATASET_VAL_DIR).name
    test_dir = output_path / Path(config.DATASET_TEST_DIR).name
    
    if not dry_run:
        train_dir.mkdir(parents=True, exist_ok=True)
        if val_ratio > 0:
            val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {
        'source_dir': str(source_dir),
        'output_base': str(output_base),
        'split_ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        },
        'random_seed': random_seed,
        'timestamp': datetime.now().isoformat(),
        'classes': {}
    }
    
    # Split each class independently to maintain balance
    print(f"\n{'─'*80}")
    print("Splitting by class...")
    print(f"{'─'*80}")
    
    for class_label, files in files_by_class.items():
        print(f"\nProcessing class {class_label} ({len(files)} files)...")
        
        # Split files
        train_files, val_files, test_files, indices = split_files(
            files, train_ratio, val_ratio, test_ratio
        )
        
        print(f"  Train: {len(train_files)} files ({len(train_files)/len(files)*100:.1f}%)")
        if val_ratio > 0:
            print(f"  Val:   {len(val_files)} files ({len(val_files)/len(files)*100:.1f}%)")
        print(f"  Test:  {len(test_files)} files ({len(test_files)/len(files)*100:.1f}%)")
        
        # Copy files if requested
        if copy_files_flag and not dry_run:
            print(f"\nCopying files for class {class_label}...")
            copy_files(train_files, train_dir, class_label, "  Train")
            if val_ratio > 0 and val_files:
                copy_files(val_files, val_dir, class_label, "  Val")
            copy_files(test_files, test_dir, class_label, "  Test")
        
        # Store statistics
        stats['classes'][class_label] = {
            'total_files': len(files),
            'train_count': len(train_files),
            'val_count': len(val_files),
            'test_count': len(test_files),
            'train_files': [f.name for f in train_files],
            'val_files': [f.name for f in val_files],
            'test_files': [f.name for f in test_files],
            'indices': indices
        }
    
    # Overall statistics
    total_files = sum(len(files) for files in files_by_class.values())
    total_train = sum(stats['classes'][c]['train_count'] for c in stats['classes'])
    total_val = sum(stats['classes'][c]['val_count'] for c in stats['classes'])
    total_test = sum(stats['classes'][c]['test_count'] for c in stats['classes'])
    
    stats['totals'] = {
        'total_files': total_files,
        'train_count': total_train,
        'val_count': total_val,
        'test_count': total_test
    }
    
    # Save split information
    if not dry_run:
        split_info_path = output_path / "split_info.json"
        with open(split_info_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n[OK] Split information saved to: {split_info_path}")
    
    return stats


def print_summary(stats):
    """Print split summary."""
    if not stats:
        return
    
    print(f"\n{'='*80}")
    print("SPLIT SUMMARY")
    print(f"{'='*80}")
    
    totals = stats['totals']
    print(f"\nTotal files: {totals['total_files']}")
    print(f"  Train: {totals['train_count']} ({totals['train_count']/totals['total_files']*100:.1f}%)")
    if totals['val_count'] > 0:
        print(f"  Val:   {totals['val_count']} ({totals['val_count']/totals['total_files']*100:.1f}%)")
    print(f"  Test:  {totals['test_count']} ({totals['test_count']/totals['total_files']*100:.1f}%)")
    
    print(f"\nPer-class breakdown:")
    for class_label, class_stats in stats['classes'].items():
        print(f"\n  Class {class_label}:")
        print(f"    Total: {class_stats['total_files']}")
        print(f"    Train: {class_stats['train_count']}")
        if class_stats['val_count'] > 0:
            print(f"    Val:   {class_stats['val_count']}")
        print(f"    Test:  {class_stats['test_count']}")
    
    print(f"\n{'='*80}")
    print("✓ Split complete! Datasets are ready for use.")
    print(f"{'='*80}\n")


def verify_no_overlap(stats):
    """Verify that there is no file overlap between splits."""
    print("\nVerifying no data leakage...")
    
    for class_label, class_stats in stats['classes'].items():
        train_set = set(class_stats['train_files'])
        val_set = set(class_stats['val_files'])
        test_set = set(class_stats['test_files'])
        
        # Check overlaps
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print(f"[ERROR] Class {class_label} has overlapping files!")
            if train_val_overlap:
                print(f"  Train-Val overlap: {len(train_val_overlap)} files")
            if train_test_overlap:
                print(f"  Train-Test overlap: {len(train_test_overlap)} files")
            if val_test_overlap:
                print(f"  Val-Test overlap: {len(val_test_overlap)} files")
            return False
        else:
            print(f"  ✓ Class {class_label}: No overlap detected")
    
    print("\n✓ Verification passed: Zero data leakage confirmed!\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/validation/test sets with no overlap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 70/15/15 split with validation
  python split_dataset.py --source dataset_combined --train 0.7 --val 0.15 --test 0.15
  
  # Simple 80/20 train/test split (no validation)
  python split_dataset.py --source dataset_combined --train 0.8 --test 0.2 --no-val
  
  # Custom 60/20/20 split
  python split_dataset.py --source dataset_combined --train 0.6 --val 0.2 --test 0.2
  
  # Dry run (preview split without copying files)
  python split_dataset.py --source dataset_combined --dry-run
  
  # Custom random seed for different split
  python split_dataset.py --source dataset_combined --seed 123
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source dataset directory (must contain 0/ and 1/ subdirectories)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output base directory (default: parent of source directory)'
    )
    
    parser.add_argument(
        '--train',
        type=float,
        default=0.7,
        help='Training set proportion (default: 0.7)'
    )
    
    parser.add_argument(
        '--val',
        type=float,
        default=0.15,
        help='Validation set proportion (default: 0.15, use 0 to skip)'
    )
    
    parser.add_argument(
        '--test',
        type=float,
        default=0.15,
        help='Test set proportion (default: 0.15)'
    )
    
    parser.add_argument(
        '--no-val',
        action='store_true',
        help='Skip validation set (sets --val to 0 and adjusts other ratios)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without copying files'
    )
    
    parser.add_argument(
        '--no-copy',
        action='store_true',
        help='Do not copy files, just generate split info'
    )
    
    args = parser.parse_args()
    
    # Handle --no-val flag
    if args.no_val:
        args.val = 0.0
        # Renormalize train and test to sum to 1.0
        total = args.train + args.test
        args.train = args.train / total
        args.test = args.test / total
    
    # Determine output directory
    if args.output is None:
        # Use parent of source directory
        args.output = str(Path(args.source).parent)
    
    # Get script directory for relative paths
    script_dir = Path(__file__).parent
    source_path = script_dir / args.source
    
    if not source_path.exists():
        print(f"[ERROR] Source directory not found: {source_path}")
        return 1
    
    # Create splits
    stats = create_dataset_splits(
        source_dir=source_path,
        output_base=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        random_seed=args.seed,
        copy_files_flag=not args.no_copy,
        dry_run=args.dry_run
    )
    
    if stats is None:
        return 1
    
    # Verify no overlap
    if not args.dry_run:
        verify_no_overlap(stats)
    
    # Print summary
    print_summary(stats)
    
    if args.dry_run:
        print("[INFO] This was a dry run. Run without --dry-run to actually create splits.\n")
    elif args.no_copy:
        print("[INFO] Split info generated without copying files.\n")
    else:
        print("[SUCCESS] Dataset split complete!\n")
        print("Next steps:")
        print("  1. Use DATASET_ROOT_OVERRIDE='dataset_train' for training")
        print("  2. Use DATASET_ROOT_OVERRIDE='dataset_test' for testing")
        print("  3. Update config.py DEFAULT_DATASET if needed\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
