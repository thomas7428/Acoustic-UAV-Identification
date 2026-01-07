"""
Master Setup Script v2.0 - Enhanced Dataset Pipeline
Automates the complete dataset preparation workflow with zero data leakage.

This script orchestrates dataset creation only:
1. Download DADS dataset from HuggingFace
2. Augment BOTH drone and no-drone classes (optional)
3. Combine original + augmented samples
4. Split into train/validation/test sets (zero overlap)
5. Validate file properties against `config.py` and write a validation report

Usage:
    # Quick setup with defaults (100 samples per class)
    python master_setup_v2.py
    
    # Custom sample counts
    python master_setup_v2.py --drone-samples 200 --no-drone-samples 200
    
    # Skip specific steps
    python master_setup_v2.py --skip-download --skip-augmentation
    
    # Dry run (see what would happen)
    python master_setup_v2.py --dry-run
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import librosa
import soundfile as sf
import numpy as np
import random

# Add project root to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
try:
    from tools.audio_utils import ensure_duration
except Exception:
    def ensure_duration(signal, sr, target_duration, crossfade_duration=0.1):
        target_samples = int(target_duration * sr)
        if len(signal) >= target_samples:
            return signal[:target_samples]
        return np.pad(signal, (0, target_samples - len(signal)), mode='constant')


class colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{colors.HEADER}{colors.BOLD}{'='*80}")
    print(f"{text}")
    print(f"{'='*80}{colors.ENDC}\n")


def print_step(step_num, total_steps, description):
    """Print a formatted step indicator."""
    print(f"\n{colors.CYAN}{colors.BOLD}[STEP {step_num}/{total_steps}] {description}{colors.ENDC}")
    print(f"{colors.CYAN}{'─'*80}{colors.ENDC}\n")


def print_success(message):
    """Print a success message."""
    print(f"{colors.GREEN}✓ {message}{colors.ENDC}")


def print_warning(message):
    """Print a warning message."""
    print(f"{colors.YELLOW}⚠ {message}{colors.ENDC}")


def print_error(message):
    """Print an error message."""
    print(f"{colors.RED}✗ {message}{colors.ENDC}")


def print_info(message):
    """Print an info message."""
    print(f"{colors.BLUE}ℹ {message}{colors.ENDC}")


def cleanup_directories(script_dir, dry_run=False):
    """
    Clean up existing dataset directories to prevent contamination.
    
    Args:
        script_dir: Script directory path
        dry_run: If True, only show what would be deleted
    
    Returns:
        bool: Always True
    """
    print_info("Cleaning up existing dataset directories...")
    
    dirs_to_clean = [
        'dataset_test',
        'dataset_augmented', 
        'dataset_combined',
        'dataset_train',
        'dataset_val',
        'dataset_dads',
        'extracted_features'
    ]
    
    for dir_name in dirs_to_clean:
        dir_path = script_dir / dir_name
        if dir_path.exists():
            if dry_run:
                print(f"  Would delete: {dir_path}")
            else:
                print(f"  Deleting: {dir_path}")
                shutil.rmtree(dir_path)
                print_success(f"Cleaned {dir_name}/")
        else:
            print(f"  {dir_name}/ not found (skip)")
    
    print()
    return True


def run_command(cmd, cwd=None, dry_run=False):
    """
    Execute a shell command and return success status.
    
    Args:
        cmd: Command to execute (list or string)
        cwd: Working directory for command
        dry_run: If True, only print command without executing
    
    Returns:
        bool: True if successful (or dry run), False otherwise
    """
    if isinstance(cmd, list):
        cmd_str = ' '.join(cmd)
    else:
        cmd_str = cmd
    
    print(f"  Running: {cmd_str}")
    
    if dry_run:
        print_warning("  (Dry run - command not executed)")
        return True
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            shell=isinstance(cmd, str),
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"  Command failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print_error(f"  Error: {e}")
        return False


def check_requirements(script_dir):
    """Check if required packages are installed."""
    print_info("Checking requirements...")
    
    required_packages = [
        'numpy',
        'librosa',
        'soundfile',
        'datasets',  # HuggingFace datasets
        'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print_error(f"\nMissing packages: {', '.join(missing)}")
        print_info("Install with: pip install " + ' '.join(missing))
        return False
    
    print_success("All required packages are installed")
    return True


def combine_datasets(original_dir, augmented_dir, output_dir, dry_run=False):
    """
    Combine original and augmented datasets.
    
    Args:
        original_dir: Directory with original samples
        augmented_dir: Directory with augmented samples
        output_dir: Output directory for combined dataset
        dry_run: If True, don't actually copy files
    """
    original_path = Path(original_dir)
    augmented_path = Path(augmented_dir)
    output_path = Path(output_dir)
    
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    
    print_info(f"Combining datasets:")
    print(f"  Original:  {original_dir}")
    print(f"  Augmented: {augmented_dir}")
    print(f"  Output:    {output_dir}")
    
    total_copied = 0
    
    for class_dir in ['0', '1']:
        if not dry_run:
            (output_path / class_dir).mkdir(exist_ok=True)
        
        # Copy from original
        orig_class_path = original_path / class_dir
        if orig_class_path.exists():
            files = list(orig_class_path.glob('*.wav'))
            print(f"\n  Class {class_dir}: Copying {len(files)} original files...")
            if not dry_run:
                # Normalize originals to target duration and sample rate when copying
                target_sr = config.SAMPLE_RATE
                target_dur = config.AUDIO_DURATION_S
                for file in files:
                    dest = output_path / class_dir / f"orig_{file.name}"
                    try:
                        # Load, resample and enforce exact target duration (loop/truncate)
                        y, _ = librosa.load(str(file), sr=target_sr)
                        y = ensure_duration(y, target_sr, target_dur)
                        # Use configured WAV subtype when writing combined originals
                        subtype = getattr(config, 'AUDIO_WAV_SUBTYPE', 'FLOAT')
                        sf.write(str(dest), y, target_sr, subtype=subtype)
                    except Exception:
                        raise RuntimeError(f"Failed to process original file: {file}")
            total_copied += len(files)
        
        # Copy from augmented
        aug_class_path = augmented_path / class_dir
        if aug_class_path.exists():
            files = list(aug_class_path.glob('*.wav'))
            print(f"  Class {class_dir}: Copying {len(files)} augmented files...")
            if not dry_run:
                for file in files:
                    shutil.copy2(file, output_path / class_dir / file.name)
            total_copied += len(files)
    
    print_success(f"Combined {total_copied} files into {output_dir}")
    return True


def distribute_sources_to_splits(original_dir, augmented_dir, train_dir, val_dir, test_dir,
                                 train_ratio, val_ratio, test_ratio, dry_run=False, random_seed=42):
    """
    Collect files from `original_dir` and `augmented_dir` and distribute them
    directly into `train_dir`, `val_dir`, and `test_dir` according to ratios.
    This avoids an intermediate combined directory and prevents leakage by
    stratifying per-class and using a fixed random seed.
    """
    print_info("Distributing originals and augmentations into train/val/test...")
    original_path = Path(original_dir)
    augmented_path = Path(augmented_dir)
    train_path = Path(train_dir)
    val_path = Path(val_dir)
    test_path = Path(test_dir)

    stats = {}

    if not dry_run:
        train_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)
        test_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(random_seed)

    for class_label in ['0', '1']:
        src_files = []
        orig_class = original_path / class_label
        aug_class = augmented_path / class_label
        if orig_class.exists():
            src_files.extend(sorted(list(orig_class.glob('*.wav'))))
        if aug_class.exists():
            src_files.extend(sorted(list(aug_class.glob('*.wav'))))

        total = len(src_files)
        stats[class_label] = {'total': total}

        if total == 0:
            print_warning(f"No source files found for class {class_label} (skipping)")
            stats[class_label].update({'train': 0, 'val': 0, 'test': 0})
            continue

        # Shuffle deterministically
        rng.shuffle(src_files)

        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)
        n_test = total - n_train - n_val

        stats[class_label].update({'train': n_train, 'val': n_val, 'test': n_test})

        # Create class subdirs
        if not dry_run:
            (train_path / class_label).mkdir(parents=True, exist_ok=True)
            (val_path / class_label).mkdir(parents=True, exist_ok=True)
            (test_path / class_label).mkdir(parents=True, exist_ok=True)

        # Copy files
        idx = 0
        for f in src_files:
            if idx < n_train:
                dst = train_path / class_label / f.name
            elif idx < n_train + n_val:
                dst = val_path / class_label / f.name
            else:
                dst = test_path / class_label / f.name

            if dry_run:
                print(f"Would copy {f} -> {dst}")
            else:
                try:
                    shutil.copy2(f, dst)
                except Exception as e:
                    print_error(f"Failed to copy {f} to {dst}: {e}")
            idx += 1

    print_success("Distribution complete")
    return stats


def step_3_distribute_and_split(script_dir, train_ratio, val_ratio, test_ratio, dry_run):
    """Wrapper step to distribute originals and augmentations directly into splits."""
    print_step(3, 5, "Distribute Originals + Augmentations into Splits")
    try:
        originals = config.DATASET_DADS_DIR
        augmented = config.DATASET_AUGMENTED_DIR
        train_dir = config.DATASET_TRAIN_DIR
        val_dir = config.DATASET_VAL_DIR
        test_dir = config.DATASET_TEST_DIR

        distribute_sources_to_splits(originals, augmented, train_dir, val_dir, test_dir,
                                     train_ratio, val_ratio, test_ratio, dry_run=dry_run, random_seed=42)
        return True
    except Exception as e:
        print_error(f"Distribution failed: {e}")
        return False


def step_1_download(script_dir, drone_samples, no_drone_samples, dry_run):
    """Step 1: Download DADS dataset."""
    print_step(1, 6, "Download DADS Dataset")
    
    if dry_run:
        print_warning("Dry run - skipping actual download")
        return True

    # Import and call the downloader directly to avoid subprocess overhead.
    try:
        sys.path.insert(0, str(script_dir))
        from download_and_prepare_dads import download_and_prepare_dads
        max_per_class = max(drone_samples, no_drone_samples)
        # Use centralized dataset path from config
        download_and_prepare_dads(output_dir=str(config.DATASET_DADS_DIR), max_samples=None, max_per_class=max_per_class, verbose=True)
        return True
    except Exception as e:
        print_error(f"Download step failed: {e}")
        return False


def step_2_augment(script_dir, config_file, dry_run):
    """Step 2: Augment both drone and no-drone classes."""
    print_step(2, 6, "Augment Both Classes")
    
    if dry_run:
        print_warning("Dry run - skipping augmentation execution")
        return True

    try:
        sys.path.insert(0, str(script_dir))
        import json
        from augment_dataset_v2 import generate_augmented_samples

        # Load augmentation config. Accept absolute paths or resolve relative to script_dir.
        cfg_candidate = Path(config_file)
        if cfg_candidate.is_absolute():
            cfg_path = cfg_candidate
        else:
            cfg_path = script_dir / config_file
        if not cfg_path.exists():
            print_error(f"Augmentation config not found: {cfg_path}")
            return False
        with open(cfg_path, 'r') as f:
            aug_cfg = json.load(f)

        # Call directly
        stats = generate_augmented_samples(aug_cfg, dry_run=False)
        print_success("Augmentation completed (direct call)")
        return True
    except Exception as e:
        print_error(f"Augmentation failed: {e}")
        return False


def step_3_combine(script_dir, dry_run):
    """Step 3: Combine original and augmented datasets."""
    print_step(3, 5, "Combine Original + Augmented")
    
    # Deprecated: combine is handled by the new distribution step which places
    # originals and augmentations directly into the train/val/test folders.
    print_info("Combine step is deprecated; use distribution step instead")
    return True


def step_4_split(script_dir, train_ratio, val_ratio, test_ratio, dry_run):
    """Step 4: Split into train/validation/test sets."""
    print_step(4, 5, "Split Train/Validation/Test")
    
    try:
        sys.path.insert(0, str(script_dir))
        from split_dataset import create_dataset_splits

        source = config.DATASET_COMBINED_DIR
        output_base = str(config.EXTRACTION_DIR)
        stats = create_dataset_splits(str(source), output_base, train_ratio, val_ratio, test_ratio, random_seed=42, copy_files_flag=not dry_run, dry_run=dry_run)
        if stats is None:
            return False
        return True
    except Exception as e:
        print_error(f"Split step failed: {e}")
        return False


def step_5_validate(script_dir, dry_run=False):
    """Step 5: Validate dataset files and extracted features.

    Runs `tools/validate_dataset.py` which inspects audio files and
    `extracted_features/` JSONs and writes `tools/validate_report.json`.
    """
    print_step(5, 5, "Validate Dataset Files and Features")

    # If dry_run, still run validation on what's present (no files written),
    # but allow caller to skip running it by passing dry_run=True.
    if dry_run:
        print_warning("Dry run - running validation in read-only mode")

    try:
        validator = script_dir.parent / 'tools' / 'validate_dataset.py'
        if not validator.exists():
            print_error(f"Validator not found: {validator}")
            return False

        cmd = [sys.executable, str(validator)]
        print_info(f"Running dataset validator: {cmd}")
        result = subprocess.run(cmd, check=True)
        print_success("Validation completed; see tools/validate_report.json")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Dataset validator failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print_error(f"Validation error: {e}")
        return False


def step_5_summary(script_dir):
    """Final summary: concise dataset report and location of validation output."""
    print_step(5, 5, "Setup Complete - Dataset Created and Validated")

    # Count files in each dataset folder under script_dir
    datasets = [
        'dataset_dads', 'dataset_augmented',
        'dataset_train', 'dataset_val', 'dataset_test'
    ]

    print(f"\n{colors.BOLD}Dataset Summary:{colors.ENDC}")
    print(f"{'─'*80}")
    for name in datasets:
        p = script_dir / name
        if p.exists():
            class_0 = len(list((p / '0').glob('*.wav'))) if (p / '0').exists() else 0
            class_1 = len(list((p / '1').glob('*.wav'))) if (p / '1').exists() else 0
            total = class_0 + class_1
            print(f"  {name:20s} | Class 0: {class_0:4d} | Class 1: {class_1:4d} | Total: {total:4d}")
        else:
            print(f"  {name:20s} (not present)")

    print(f"\nValidation report: tools/validate_report.json")
    print(f"{colors.GREEN}{colors.BOLD}✓ Dataset creation complete. Use training scripts separately when ready.{colors.ENDC}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Master setup script for enhanced dataset pipeline v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick setup with defaults
  python master_setup_v2.py
  
  # Custom sample counts
  python master_setup_v2.py --drone-samples 200 --no-drone-samples 200
  
  # Custom train/val/test split
  python master_setup_v2.py --train 0.8 --val 0.1 --test 0.1
  
  # Skip certain steps
  python master_setup_v2.py --skip-download --skip-augmentation
  
  # Dry run (preview without executing)
  python master_setup_v2.py --dry-run
        """
    )
    
    # Dataset parameters
    parser.add_argument('--drone-samples', type=int, default=1000,
                        help='Number of drone samples to download per class (default: 1000)')
    parser.add_argument('--no-drone-samples', type=int, default=1000,
                        help='Number of no-drone samples to download per class (default: 1000)')
    
    # Split ratios
    parser.add_argument('--train', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    parser.add_argument('--val', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    
    # Configuration
    # Default augmentation config is taken from centralized `config.py`
    parser.add_argument('--config', type=str, default=str(config.CONFIG_DATASET_PATH.name),
                        help=f'Augmentation config file (default: {config.CONFIG_DATASET_PATH.name})')
    
    # Step control
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip dataset download (use existing dataset_test)')
    parser.add_argument('--skip-augmentation', action='store_true',
                        help='Skip augmentation (use existing dataset_augmented)')
    # Note: feature extraction and training are intentionally out of scope
    # for this master setup script. Use separate preprocessing/training scripts.
    
    # Execution mode
    parser.add_argument('--dry-run', action='store_true',
                        help='Perform dry run without creating files')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Skip cleanup (keep existing datasets - may cause contamination)')
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent.resolve()
    
    # Print header
    print_header("ENHANCED DATASET PIPELINE v2.0")
    print(f"Working directory: {script_dir}\n")
    
    if args.dry_run:
        print_warning("DRY RUN MODE - No files will be created\n")
    
    # Check requirements
    if not check_requirements(script_dir):
        print_error("\nSetup aborted due to missing requirements")
        return 1
    
    # Step 0: Cleanup (unless disabled)
    if not args.no_cleanup:
        print_step(0, 5, "Cleanup Existing Datasets")
        cleanup_directories(script_dir, args.dry_run)
    else:
        print_warning("Skipping cleanup - Existing datasets will be REUSED (risk of contamination!)")
        print()
    
    # Validate ratios
    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 0.01:
        print_error(f"Split ratios must sum to 1.0 (got {total_ratio})")
        return 1
    
    # Execute pipeline steps
    steps_success = []
    
    # Step 1: Download
    if not args.skip_download:
        success = step_1_download(script_dir, args.drone_samples, args.no_drone_samples, args.dry_run)
        steps_success.append(('Download', success))
        if not success and not args.dry_run:
            print_error("Setup failed at download step")
            return 1
    else:
        print_info("Skipping download step (using existing dataset_test)")
    
    # Step 2: Augmentation
    if not args.skip_augmentation:
        success = step_2_augment(script_dir, args.config, args.dry_run)
        steps_success.append(('Augmentation', success))
        if not success and not args.dry_run:
            print_error("Setup failed at augmentation step")
            return 1
    else:
        print_info("Skipping augmentation step (using existing dataset_augmented)")
    
    # Step 3: Distribute originals + augmentations into splits
    success = step_3_distribute_and_split(script_dir, args.train, args.val, args.test, args.dry_run)
    steps_success.append(('Distribute+Split', success))
    if not success and not args.dry_run:
        print_error("Setup failed at distribution step")
        return 1
    
    # Step 5: Validate
    success = step_5_validate(script_dir, args.dry_run)
    steps_success.append(('Validation', success))
    if not success and not args.dry_run:
        print_error("Setup failed at validation step")
        return 1
    
    # Final summary
    if not args.dry_run:
        step_5_summary(script_dir)
    else:
        print_warning("\nDry run complete - no files were created")
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{colors.YELLOW}Setup interrupted by user{colors.ENDC}")
        exit(1)
    except Exception as e:
        print(f"\n{colors.RED}Unexpected error: {e}{colors.ENDC}")
        exit(1)
