"""
Master Setup Script v2.0 - Enhanced Dataset Pipeline
Automates the complete dataset preparation workflow with zero data leakage.

This script orchestrates:
1. Download DADS dataset from HuggingFace
2. Augment BOTH drone and no-drone classes
3. Combine original + augmented samples
4. Split into train/validation/test sets (zero overlap)
5. Extract Mel spectrogram features for training

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
import config
import config


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
                        # Load, resample and pad/truncate to target duration
                        y, _ = librosa.load(str(file), sr=target_sr)
                        target_samples = int(target_sr * float(target_dur))
                        if len(y) > target_samples:
                            y = y[:target_samples]
                        elif len(y) < target_samples:
                            y = np.pad(y, (0, target_samples - len(y)), mode='constant')
                        sf.write(str(dest), y, target_sr)
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


def step_1_download(script_dir, drone_samples, no_drone_samples, dry_run):
    """Step 1: Download DADS dataset."""
    print_step(1, 6, "Download DADS Dataset")
    
    if dry_run:
        print_warning("Dry run - skipping actual download")
        return True
    
    cmd = [
        sys.executable,
        "download_and_prepare_dads.py",
        "--output", "dataset_dads",
        "--max-per-class", str(max(drone_samples, no_drone_samples))
    ]
    
    return run_command(cmd, cwd=script_dir, dry_run=dry_run)


def step_2_augment(script_dir, config_file, dry_run):
    """Step 2: Augment both drone and no-drone classes."""
    print_step(2, 6, "Augment Both Classes")
    
    cmd = [
        sys.executable,
        "augment_dataset_v2.py",
        "--config", config_file
    ]
    
    if dry_run:
        cmd.append("--dry-run")
    
    return run_command(cmd, cwd=script_dir, dry_run=dry_run)


def step_3_combine(script_dir, dry_run):
    """Step 3: Combine original and augmented datasets."""
    print_step(3, 6, "Combine Original + Augmented")
    
    return combine_datasets(
        original_dir=script_dir / "dataset_test",
        augmented_dir=script_dir / "dataset_augmented",
        output_dir=script_dir / "dataset_combined",
        dry_run=dry_run
    )


def step_4_split(script_dir, train_ratio, val_ratio, test_ratio, dry_run):
    """Step 4: Split into train/validation/test sets."""
    print_step(4, 6, "Split Train/Validation/Test")
    
    cmd = [
        sys.executable,
        "split_dataset.py",
        "--source", "dataset_combined",
        "--output", str(script_dir),
        "--train", str(train_ratio),
        "--val", str(val_ratio),
        "--test", str(test_ratio)
    ]
    
    if dry_run:
        cmd.append("--dry-run")
    
    return run_command(cmd, cwd=script_dir, dry_run=dry_run)


def step_5_extract_features(script_dir, dry_run):
    """Step 5: Extract Mel spectrogram features."""
    print_step(5, 6, "Extract Mel Spectrogram Features")
    
    if dry_run:
        print_warning("Dry run - skipping feature extraction")
        return True
    
    # Set environment variable to use training dataset
    env = os.environ.copy()
    env['DATASET_ROOT_OVERRIDE'] = 'dataset_train'
    
    cmd = [
        sys.executable,
        str(script_dir.parent / "1 - Preprocessing and Features Extraction" / "Mel_Preprocess_and_Feature_Extract.py")
    ]
    
    print_info("Using dataset_train for feature extraction")
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Feature extraction failed with exit code {e.returncode}")
        return False


def step_6_summary(script_dir):
    """Step 6: Print summary and next steps."""
    print_step(6, 6, "Setup Complete!")
    
    print_success("Dataset pipeline completed successfully!\n")
    
    # Count files in each dataset
    datasets = {
        'dataset_test': 'Original DADS samples',
        'dataset_augmented': 'Augmented samples',
        'dataset_combined': 'Combined (originals + augmented)',
        'dataset_train': 'Training set',
        'dataset_val': 'Validation set',
        'dataset_test': 'Test set (final evaluation)'
    }
    
    print(f"\n{colors.BOLD}Dataset Summary:{colors.ENDC}")
    print(f"{'─'*80}")
    
    for dataset_name, description in datasets.items():
        dataset_path = script_dir / dataset_name
        if dataset_path.exists():
            class_0 = len(list((dataset_path / '0').glob('*.wav'))) if (dataset_path / '0').exists() else 0
            class_1 = len(list((dataset_path / '1').glob('*.wav'))) if (dataset_path / '1').exists() else 0
            total = class_0 + class_1
            print(f"  {dataset_name:25s} {description:30s} | Class 0: {class_0:4d} | Class 1: {class_1:4d} | Total: {total:4d}")
    
    print(f"\n{colors.BOLD}Next Steps:{colors.ENDC}")
    print(f"{'─'*80}")
    print(f"  1. Train models:")
    print(f"     {colors.CYAN}set DATASET_ROOT_OVERRIDE=dataset_train{colors.ENDC}")
    print(f"     python \"2 - Model Training\\CNN_Trainer.py\"")
    print(f"\n  2. Evaluate on test set:")
    print(f"     {colors.CYAN}set DATASET_ROOT_OVERRIDE=dataset_test{colors.ENDC}")
    print(f"     python \"3 - Single Model Performance Calculation\\CNN_and_CRNN_Performance_Calcs.py\"")
    print(f"\n  3. Run visualizations:")
    print(f"     python \"6 - Visualization\\performance_by_distance.py\"")
    
    print(f"\n{colors.BOLD}Key Files:{colors.ENDC}")
    print(f"{'─'*80}")
    
    key_files = [
        ("split_info.json", "Dataset split details and verification"),
        ("dataset_augmented/augmentation_metadata.json", "Augmentation metadata"),
        ("extracted_features/mel_pitch_shift_9.0.json", "Extracted Mel features")
    ]
    
    for filename, description in key_files:
        file_path = script_dir / filename
        status = "✓" if file_path.exists() else "✗"
        print(f"  {status} {filename:45s} {description}")
    
    print(f"\n{colors.GREEN}{colors.BOLD}✓ Setup complete! Your datasets are ready for training.{colors.ENDC}\n")


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
    parser.add_argument('--drone-samples', type=int, default=400,
                        help='Number of drone samples to download per class (default: 400)')
    parser.add_argument('--no-drone-samples', type=int, default=400,
                        help='Number of no-drone samples to download per class (default: 400)')
    
    # Split ratios
    parser.add_argument('--train', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    parser.add_argument('--val', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    
    # Configuration
    parser.add_argument('--config', type=str, default='augment_config_v2.json',
                        help='Augmentation config file (default: augment_config_v2.json)')
    
    # Step control
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip dataset download (use existing dataset_test)')
    parser.add_argument('--skip-augmentation', action='store_true',
                        help='Skip augmentation (use existing dataset_augmented)')
    parser.add_argument('--skip-features', action='store_true',
                        help='Skip feature extraction')
    
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
        print_step(0, 6, "Cleanup Existing Datasets")
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
    
    # Step 3: Combine
    success = step_3_combine(script_dir, args.dry_run)
    steps_success.append(('Combine', success))
    if not success and not args.dry_run:
        print_error("Setup failed at combine step")
        return 1
    
    # Step 4: Split
    success = step_4_split(script_dir, args.train, args.val, args.test, args.dry_run)
    steps_success.append(('Split', success))
    if not success and not args.dry_run:
        print_error("Setup failed at split step")
        return 1
    
    # Step 5: Feature extraction
    if not args.skip_features:
        success = step_5_extract_features(script_dir, args.dry_run)
        steps_success.append(('Feature Extraction', success))
        if not success and not args.dry_run:
            print_error("Setup failed at feature extraction step")
            return 1
    else:
        print_info("Skipping feature extraction step")
    
    # Step 6: Summary
    if not args.dry_run:
        step_6_summary(script_dir)
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
