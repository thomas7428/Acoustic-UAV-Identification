#!/usr/bin/env python3
"""
Verification script for new model integration
Checks that all files and configurations are in place before training
"""

import sys
from pathlib import Path
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def print_status(check_name, passed, message=""):
    """Print colored status message"""
    if passed:
        print(f"✅ {check_name}")
    else:
        print(f"❌ {check_name}")
        if message:
            print(f"   → {message}")
    return passed

def main():
    print("\n" + "="*70)
    print("VERIFICATION: New SOTA Models Integration")
    print("="*70 + "\n")
    
    all_passed = True
    
    # Check 1: Trainer files exist
    print("📁 Checking trainer files...")
    trainers = [
        "2 - Model Training/EfficientNet_Trainer.py",
        "2 - Model Training/MobileNet_Trainer.py",
        "2 - Model Training/Conformer_Trainer.py",
        "2 - Model Training/TCN_Trainer.py"
    ]
    
    for trainer in trainers:
        trainer_path = PROJECT_ROOT / trainer
        passed = print_status(
            f"  {trainer_path.name}",
            trainer_path.exists(),
            f"File not found: {trainer_path}"
        )
        all_passed &= passed
    
    print()
    
    # Check 2: config.py has new model paths
    print("⚙️  Checking config.py...")
    try:
        import config
        model_paths = [
            ('EFFICIENTNET_MODEL_PATH', config.EFFICIENTNET_MODEL_PATH),
            ('MOBILENET_MODEL_PATH', config.MOBILENET_MODEL_PATH),
            ('CONFORMER_MODEL_PATH', config.CONFORMER_MODEL_PATH),
            ('TCN_MODEL_PATH', config.TCN_MODEL_PATH)
        ]
        
        for name, path in model_paths:
            passed = print_status(
                f"  {name}",
                hasattr(config, name.split('_')[0] + '_MODEL_PATH'),
                f"Missing in config.py"
            )
            all_passed &= passed
    except Exception as e:
        print_status("  config.py import", False, str(e))
        all_passed = False
    
    print()
    
    # Check 3: Universal_Perf_Tester supports new models
    print("🧪 Checking Universal_Perf_Tester.py...")
    tester_path = PROJECT_ROOT / "3 - Single Model Performance Calculation/Universal_Perf_Tester.py"
    if tester_path.exists():
        content = tester_path.read_text()
        new_models = ['EFFICIENTNET', 'MOBILENET', 'CONFORMER', 'TCN']
        
        for model in new_models:
            passed = print_status(
                f"  {model} in choices",
                f"'{model}'" in content or f'"{model}"' in content,
                "Not found in argparse choices"
            )
            all_passed &= passed
    else:
        print_status("  Universal_Perf_Tester.py", False, "File not found")
        all_passed = False
    
    print()
    
    # Check 4: calibrate_thresholds supports new models
    print("🎯 Checking calibrate_thresholds.py...")
    calib_path = PROJECT_ROOT / "3 - Single Model Performance Calculation/calibrate_thresholds.py"
    if calib_path.exists():
        content = calib_path.read_text()
        
        for model in new_models:
            passed = print_status(
                f"  {model} in choices",
                f"'{model}'" in content or f'"{model}"' in content,
                "Not found in argparse choices"
            )
            all_passed &= passed
    else:
        print_status("  calibrate_thresholds.py", False, "File not found")
        all_passed = False
    
    print()
    
    # Check 5: run_full_pipeline.sh
    print("🚀 Checking run_full_pipeline.sh...")
    pipeline_path = PROJECT_ROOT / "run_full_pipeline.sh"
    if pipeline_path.exists():
        content = pipeline_path.read_text()
        
        trainer_cases = [
            'EfficientNet_Trainer.py',
            'MobileNet_Trainer.py',
            'Conformer_Trainer.py',
            'TCN_Trainer.py'
        ]
        
        for trainer_file in trainer_cases:
            passed = print_status(
                f"  {trainer_file}",
                trainer_file in content,
                "Not found in train_single_model() function"
            )
            all_passed &= passed
        
        # Check default models list
        passed = print_status(
            "  Default MODELS includes new models",
            'EFFICIENTNET' in content and 'MOBILENET' in content,
            "MODELS variable may need update"
        )
        all_passed &= passed
    else:
        print_status("  run_full_pipeline.sh", False, "File not found")
        all_passed = False
    
    print()
    
    # Check 6: Feature files exist
    print("📊 Checking MEL feature files...")
    feature_dir = PROJECT_ROOT / "0 - DADS dataset extraction/extracted_features"
    feature_files = [
        "mel_train.npz",
        "mel_val.npz",
        "mel_test.npz"
    ]
    
    for feature_file in feature_files:
        feature_path = feature_dir / feature_file
        passed = print_status(
            f"  {feature_file}",
            feature_path.exists(),
            "Run feature extraction first"
        )
        # Don't fail if features don't exist yet (can be extracted)
    
    print()
    
    # Summary
    print("="*70)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Ready to train new models!")
        print()
        print("Next steps:")
        print("  1. Train specific model:")
        print("     ./run_full_pipeline.sh --models EFFICIENTNET")
        print()
        print("  2. Train all new models:")
        print("     ./run_full_pipeline.sh --models EFFICIENTNET,MOBILENET,CONFORMER,TCN")
        print()
        print("  3. Train all 8 models:")
        print("     ./run_full_pipeline.sh --parallel")
        print()
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues before training")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
