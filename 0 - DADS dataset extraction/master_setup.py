"""
Master setup script for the Acoustic UAV Identification project.
This script unifies all setup operations in a single, easy-to-use interface.

Features:
- Clean/reset project data
- Download and prepare dataset
- Fix preprocessing compatibility
- Setup project paths
- Complete workflow automation

Usage:
    python master_setup.py --help
"""

import os
import sys
import argparse
import shutil
import subprocess
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text:^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.END}\n")


def print_step(step_num, text):
    """Print a formatted step."""
    print(f"{Colors.BOLD}{Colors.CYAN}[Step {step_num}]{Colors.END} {text}")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def get_project_paths():
    """Get important project paths."""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    
    return {
        'script_dir': script_dir,
        'project_root': project_root,
        'dataset_test': script_dir / 'dataset_test',
        'dataset_augmented': script_dir / 'dataset_augmented',
        'dataset_combined': script_dir / 'dataset_combined',
        'extracted_features': script_dir / 'extracted_features',
        'saved_models': project_root / 'saved_models',
        'results': project_root / 'results',
        'dataset_config': project_root / 'dataset_config.py',
    }


def clean_data(args):
    """Clean all generated data and models."""
    print_header("CLEANING PROJECT DATA")
    
    paths = get_project_paths()
    items_to_clean = []
    
    if args.all or args.dataset:
        items_to_clean.append(('Dataset (test)', paths['dataset_test']))
        items_to_clean.append(('Dataset (augmented)', paths['dataset_augmented']))
        items_to_clean.append(('Dataset (combined)', paths['dataset_combined']))
    
    if args.all or args.features:
        items_to_clean.append(('Extracted features', paths['extracted_features']))
    
    if args.all or args.models:
        items_to_clean.append(('Saved models', paths['saved_models']))
    
    if args.all or args.results:
        items_to_clean.append(('Results', paths['results']))
    
    if args.all or args.config:
        items_to_clean.append(('Configuration file', paths['dataset_config']))
    
    if not items_to_clean:
        print_warning("No items specified for cleaning. Use --all or specific flags.")
        return False
    
    # Show what will be cleaned
    print("The following items will be removed:")
    for name, path in items_to_clean:
        status = "✓ exists" if path.exists() else "✗ not found"
        print(f"  - {name:20s} : {path} [{status}]")
    
    # Confirm unless --force is used
    if not args.force:
        response = input(f"\n{Colors.YELLOW}Continue with cleanup? (yes/no): {Colors.END}")
        if response.lower() not in ['yes', 'y']:
            print_warning("Cleanup cancelled.")
            return False
    
    # Perform cleanup
    print()
    cleaned_count = 0
    for name, path in items_to_clean:
        try:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                print_success(f"Removed: {name}")
                cleaned_count += 1
            else:
                print_warning(f"Not found: {name}")
        except Exception as e:
            print_error(f"Failed to remove {name}: {e}")
    
    print(f"\n{Colors.GREEN}Cleanup complete! Removed {cleaned_count} item(s).{Colors.END}")
    return True


def run_script(script_name, args_list=None, description=""):
    """Run a Python script with arguments."""
    if description:
        print(f"\n{Colors.BLUE}► {description}{Colors.END}")
    
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        print_error(f"Script not found: {script_name}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    if args_list:
        cmd.extend(args_list)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print_error(f"Script failed with exit code {e.returncode}")
        return False


def augment_dataset(args):
    """Generate augmented dataset with background noise mixing."""
    print_header("DATASET AUGMENTATION")
    
    script_args = []
    if args.augment_config:
        script_args.extend(['--config', args.augment_config])
    
    return run_script(
        'augment_dataset.py',
        script_args,
        "Generating augmented drone samples with background noise..."
    )


def combine_datasets(args):
    """Combine original and augmented datasets."""
    print_header("COMBINING DATASETS")
    
    paths = get_project_paths()
    dataset_test = paths['dataset_test']
    dataset_augmented = paths['dataset_augmented']
    dataset_combined = paths['dataset_combined']
    
    # Check if source datasets exist
    if not dataset_test.exists():
        print_error(f"Original dataset not found: {dataset_test}")
        return False
    
    if not dataset_augmented.exists():
        print_warning(f"Augmented dataset not found: {dataset_augmented}")
        print_warning("Run with --augment flag to generate augmented data")
        return False
    
    # Create combined dataset directory
    os.makedirs(dataset_combined, exist_ok=True)
    
    # Combine for each label (0 and 1)
    for label in ['0', '1']:
        label_dir = dataset_combined / label
        os.makedirs(label_dir, exist_ok=True)
        
        # Copy from original dataset
        src_dir = dataset_test / label
        if src_dir.exists():
            files = list(src_dir.glob('*.wav'))
            print(f"  Copying {len(files)} files from original dataset (label {label})...")
            for src_file in files:
                dst_file = label_dir / f"orig_{src_file.name}"
                shutil.copy2(src_file, dst_file)
        
        # Copy from augmented dataset
        aug_dir = dataset_augmented / label
        if aug_dir.exists():
            files = list(aug_dir.glob('*.wav'))
            print(f"  Copying {len(files)} files from augmented dataset (label {label})...")
            for src_file in files:
                dst_file = label_dir / f"aug_{src_file.name}"
                shutil.copy2(src_file, dst_file)
    
    # Count total files
    total_0 = len(list((dataset_combined / '0').glob('*.wav')))
    total_1 = len(list((dataset_combined / '1').glob('*.wav')))
    
    print_success(f"Combined dataset created with {total_0 + total_1} total files")
    print(f"  - Label 0 (no drone): {total_0} files")
    print(f"  - Label 1 (drone): {total_1} files")
    
    return True


def download_dataset(args):
    """Download and prepare the dataset."""
    print_header("DATASET DOWNLOAD AND PREPARATION")
    
    script_args = ['--output', args.dataset_dir]
    
    if args.max_samples:
        script_args.extend(['--max-samples', str(args.max_samples)])
    
    if args.max_per_class:
        script_args.extend(['--max-per-class', str(args.max_per_class)])
    
    if args.quiet:
        script_args.append('--quiet')
    
    return run_script(
        'download_and_prepare_dads.py',
        script_args,
        "Downloading dataset from Hugging Face..."
    )


def fix_preprocessing():
    """Fix preprocessing scripts for Windows compatibility."""
    print_header("FIXING PREPROCESSING COMPATIBILITY")
    
    # Define the preprocessing scripts to fix
    preprocessing_dir = Path(__file__).parent.parent / "1 - Preprocessing and Features Extraction"
    scripts = [
        preprocessing_dir / "Mel_Preprocess_and_Feature_Extract.py",
        preprocessing_dir / "MFCC_Preprocess_and_Feature_Extract.py"
    ]
    
    fixed_count = 0
    for script_path in scripts:
        if not script_path.exists():
            print(f"  ⚠ Script not found: {script_path.name}")
            continue
        
        # Read the script
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already fixed
        if 'os.sep' in content and 'split(os.sep)' in content:
            print(f"  ✓ Already fixed: {script_path.name}")
            fixed_count += 1
            continue
        
        # Apply the fix
        original_content = content
        content = content.replace('.split("/")', '.split(os.sep)')
        content = content.replace(".split('/')", '.split(os.sep)')
        
        # Only write if changes were made
        if content != original_content:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ Fixed: {script_path.name}")
            fixed_count += 1
        else:
            print(f"  ✓ No changes needed: {script_path.name}")
            fixed_count += 1
    
    return fixed_count > 0



def setup_paths(args):
    """Setup project paths and create compatibility files."""
    print_header("CONFIGURING PROJECT PATHS")
    
    paths = get_project_paths()
    
    # Create necessary directories
    os.makedirs(paths['extracted_features'], exist_ok=True)
    os.makedirs(paths['saved_models'], exist_ok=True)
    os.makedirs(paths['results'], exist_ok=True)
    
    print_success(f"Created directories")
    
    # Create compatibility links for training scripts
    create_compatibility_files(paths)
    print_success(f"Created compatibility files")
    
    return True


def create_compatibility_files(paths):
    """Create compatibility files for training scripts that expect specific filenames."""
    
    extracted_features = paths['extracted_features']
    
    # Mapping of target files (what trainers expect) to source files (what we have)
    compatibility_files = {
        'mel_pitch_shift_9.0.json': 'mel_data.json',
        'mfcc_pitch_shift_9.0.json': 'mfcc_data.json',
    }
    
    for target_name, source_name in compatibility_files.items():
        source_path = extracted_features / source_name
        target_path = extracted_features / target_name
        
        # Create a placeholder file that will be replaced after feature extraction
        # or copy if source already exists
        if source_path.exists():
            shutil.copy2(source_path, target_path)
            print(f"  ✓ Copied {source_name} → {target_name}")
        else:
            # Create placeholder that explains the file will be created
            placeholder = {
                "info": f"This file will be created automatically when you run the feature extraction script",
                "source": source_name,
                "mapping": [],
                "mel" if "mel" in target_name else "mfcc": [],
                "labels": []
            }
            import json
            with open(target_path, 'w') as f:
                json.dump(placeholder, f, indent=4)
            print(f"  ⚠ Created placeholder for {target_name}")


def complete_setup(args):
    """Run the complete setup workflow."""
    print_header("COMPLETE PROJECT SETUP")
    
    steps = [
        "1. Download and prepare dataset",
        "2. Setup project paths and configuration",
        "3. Fix preprocessing scripts compatibility"
    ]
    
    if args.augment:
        steps.append("4. Generate augmented dataset")
        steps.append("5. Combine original and augmented datasets")
    
    print(f"{Colors.BOLD}This will perform the following steps:{Colors.END}")
    for step in steps:
        print(f"  {step}")
    print()
    
    # Step 1: Download dataset
    print_step(1, "Downloading dataset...")
    if not download_dataset(args):
        print_error("Failed to download dataset")
        return False
    print_success("Dataset downloaded and prepared")
    
    # Step 2: Setup paths (creates config and compatibility files)
    print_step(2, "Setting up project paths...")
    if not setup_paths(args):
        print_error("Failed to setup paths")
        return False
    print_success("Project paths configured")
    
    # Step 3: Fix preprocessing
    print_step(3, "Fixing preprocessing scripts...")
    if not fix_preprocessing():
        print_error("Failed to fix preprocessing scripts")
        return False
    print_success("Preprocessing scripts fixed")
    
    # Optional: Augmentation and fusion
    if args.augment:
        # Step 4: Generate augmented dataset
        print_step(4, "Generating augmented dataset...")
        if not augment_dataset(args):
            print_error("Failed to generate augmented dataset")
            return False
        print_success("Augmented dataset generated")
        
        # Step 5: Combine datasets
        print_step(5, "Combining datasets...")
        if not combine_datasets(args):
            print_error("Failed to combine datasets")
            return False
        print_success("Datasets combined")
    
    # Final step: Update compatibility files
    final_step = 6 if args.augment else 4
    print_step(final_step, "Finalizing setup...")
    create_post_setup_helper()
    print_success("Setup finalized")
    
    # Summary
    print_header("SETUP COMPLETE!")
    print(f"{Colors.GREEN}✓ All setup steps completed successfully!{Colors.END}\n")
    
    if args.augment:
        print(f"{Colors.BOLD}Dataset Information:{Colors.END}")
        print("  ✓ Original dataset: dataset_test")
        print("  ✓ Augmented dataset: dataset_augmented (1000 samples)")
        print("  ✓ Combined dataset: dataset_combined (ready for training)")
        print()
    
    print(f"{Colors.BOLD}Next steps:{Colors.END}")
    print("  1. Extract features:")
    print('     python "../1 - Preprocessing and Features Extraction/Mel_Preprocess_and_Feature_Extract.py"')
    print('     python "../1 - Preprocessing and Features Extraction/MFCC_Preprocess_and_Feature_Extract.py"')
    print()
    print("  2. After feature extraction, run:")
    print('     python "0 - DADS dataset extraction/update_training_files.py"')
    print()
    print("  3. Then train models:")
    print('     python "../2 - Model Training/CNN_Trainer.py"')
    print('     python "../2 - Model Training/RNN_Trainer.py"')
    print('     python "../2 - Model Training/CRNN_Trainer.py"')
    print()
    
    return True


def create_post_setup_helper():
    """Create a helper script to update training files after feature extraction."""
    
    script_dir = Path(__file__).parent
    helper_script = script_dir / "update_training_files.py"
    
    helper_content = '''"""
Helper script to update compatibility files after feature extraction.
Run this after extracting Mel and MFCC features.
"""

import os
import shutil
from pathlib import Path

def update_files():
    script_dir = Path(__file__).parent
    features_dir = script_dir / "extracted_features"
    
    updates = {
        'mel_pitch_shift_9.0.json': 'mel_data.json',
        'mfcc_pitch_shift_9.0.json': 'mfcc_data.json',
    }
    
    print("Updating training compatibility files...")
    for target, source in updates.items():
        source_path = features_dir / source
        target_path = features_dir / target
        
        if source_path.exists():
            shutil.copy2(source_path, target_path)
            print(f"  ✓ Updated {target}")
        else:
            print(f"  ✗ Source not found: {source}")
    
    print("\\nDone! You can now run the training scripts.")

if __name__ == "__main__":
    update_files()
'''
    
    with open(helper_script, 'w', encoding='utf-8') as f:
        f.write(helper_content)
    
    print(f"  ✓ Created update_training_files.py")


def main():
    parser = argparse.ArgumentParser(
        description='Master setup script for Acoustic UAV Identification project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete setup with default settings (50 samples per class)
  python master_setup.py --complete --max-per-class 50
  
  # Clean everything and start fresh
  python master_setup.py --clean --all --force
  python master_setup.py --complete --max-per-class 100
  
  # Clean only features and results
  python master_setup.py --clean --features --results
  
  # Download dataset only
  python master_setup.py --download --max-per-class 200
  
  # Fix preprocessing only
  python master_setup.py --fix-preprocessing
  
  # Setup paths only
  python master_setup.py --setup-paths --dataset dataset_full
        """
    )
    
    # Main commands
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument('--clean', action='store_true',
                         help='Clean generated data and models')
    commands.add_argument('--download', action='store_true',
                         help='Download and prepare dataset')
    commands.add_argument('--augment-only', action='store_true',
                         help='Generate augmented dataset only')
    commands.add_argument('--combine-only', action='store_true',
                         help='Combine existing datasets only')
    commands.add_argument('--fix-preprocessing', action='store_true',
                         help='Fix preprocessing scripts compatibility')
    commands.add_argument('--setup-paths', action='store_true',
                         help='Setup project paths and configuration')
    commands.add_argument('--complete', action='store_true',
                         help='Run complete setup workflow')
    
    # Clean options
    clean_group = parser.add_argument_group('clean options')
    clean_group.add_argument('--all', action='store_true',
                            help='Clean all data (dataset, features, models, results, config)')
    clean_group.add_argument('--dataset', action='store_true',
                            help='Clean dataset only')
    clean_group.add_argument('--features', action='store_true',
                            help='Clean extracted features only')
    clean_group.add_argument('--models', action='store_true',
                            help='Clean saved models only')
    clean_group.add_argument('--results', action='store_true',
                            help='Clean results only')
    clean_group.add_argument('--config', action='store_true',
                            help='Clean configuration file only')
    clean_group.add_argument('--force', action='store_true',
                            help='Skip confirmation prompt')
    
    # Dataset options
    dataset_group = parser.add_argument_group('dataset options')
    dataset_group.add_argument('--dataset-dir', type=str, default='dataset_test',
                              help='Dataset directory name (default: dataset_test)')
    dataset_group.add_argument('--max-samples', type=int,
                              help='Maximum total samples to download')
    dataset_group.add_argument('--max-per-class', type=int,
                              help='Maximum samples per class')
    dataset_group.add_argument('--augment', action='store_true',
                              help='Generate augmented dataset and combine with original')
    dataset_group.add_argument('--augment-config', type=str, default='augment_config.json',
                              help='Path to augmentation config file (default: augment_config.json)')
    dataset_group.add_argument('--pitch-shift', action='store_true',
                              help='Configure to use pitch-shifted data')
    dataset_group.add_argument('--quiet', action='store_true',
                              help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Execute command
    try:
        if args.clean:
            success = clean_data(args)
        elif args.download:
            success = download_dataset(args)
        elif args.augment_only:
            success = augment_dataset(args)
        elif args.combine_only:
            success = combine_datasets(args)
        elif args.fix_preprocessing:
            success = fix_preprocessing()
        elif args.setup_paths:
            success = setup_paths(args)
        elif args.complete:
            success = complete_setup(args)
        else:
            parser.print_help()
            success = False
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Operation cancelled by user{Colors.END}")
        return 130
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
