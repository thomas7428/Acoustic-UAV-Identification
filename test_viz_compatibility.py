#!/usr/bin/env python3
"""
Test Visualization Compatibility
Validates that all visualization scripts are compatible with:
- New distance categories (600m, 700m)
- Attention-CRNN model
- Config v3 structure
"""

import sys
from pathlib import Path

# Import config
import config

def test_config_paths():
    """Test that all required paths are defined in config."""
    print("=" * 70)
    print("TEST 1: Config Paths Validation")
    print("=" * 70)
    
    required_paths = {
        'CNN_MODEL_PATH': config.CNN_MODEL_PATH,
        'RNN_MODEL_PATH': config.RNN_MODEL_PATH,
        'CRNN_MODEL_PATH': config.CRNN_MODEL_PATH,
        'ATTENTION_CRNN_MODEL_PATH': config.ATTENTION_CRNN_MODEL_PATH,
        'CNN_SCORES_PATH': config.CNN_SCORES_PATH,
        'RNN_SCORES_PATH': config.RNN_SCORES_PATH,
        'CRNN_SCORES_PATH': config.CRNN_SCORES_PATH,
        'ATTENTION_CRNN_SCORES_PATH': config.ATTENTION_CRNN_SCORES_PATH,
        'CNN_ACC_PATH': config.CNN_ACC_PATH,
        'RNN_ACC_PATH': config.RNN_ACC_PATH,
        'CRNN_ACC_PATH': config.CRNN_ACC_PATH,
        'ATTENTION_CRNN_ACC_PATH': config.ATTENTION_CRNN_ACC_PATH,
    }
    
    all_ok = True
    for name, path in required_paths.items():
        status = "✓" if path else "✗"
        print(f"  {status} {name}: {path}")
        if not path:
            all_ok = False
    
    if all_ok:
        print("\n✅ All config paths defined!")
    else:
        print("\n❌ Some config paths missing!")
    
    return all_ok


def test_visualization_imports():
    """Test that visualization scripts can be imported."""
    print("\n" + "=" * 70)
    print("TEST 2: Visualization Script Imports")
    print("=" * 70)
    
    viz_scripts = [
        'dataset_analysis',
        'model_performance',
        'augmentation_impact',
        'performance_by_distance',
        'calibrate_thresholds',
    ]
    
    all_ok = True
    sys.path.insert(0, str(Path(__file__).parent / "6 - Visualization"))
    
    for script_name in viz_scripts:
        try:
            __import__(script_name)
            print(f"  ✓ {script_name}.py")
        except Exception as e:
            print(f"  ✗ {script_name}.py - {e}")
            all_ok = False
    
    if all_ok:
        print("\n✅ All visualization scripts importable!")
    else:
        print("\n❌ Some visualization scripts have import errors!")
    
    return all_ok


def test_config_v3_structure():
    """Test config v3 JSON structure."""
    print("\n" + "=" * 70)
    print("TEST 3: Config v3 Structure Validation")
    print("=" * 70)
    
    import json
    
    config_path = Path("0 - DADS dataset extraction/augment_config_v3_balanced.json")
    
    if not config_path.exists():
        print(f"  ✗ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Check required sections
        required_sections = [
            'version',
            'drone_augmentation',
            'no_drone_augmentation',
            'audio_parameters',
            'output',
            'source_datasets',
            'advanced',
            'advanced_augmentations',
        ]
        
        all_ok = True
        for section in required_sections:
            if section in config_data:
                print(f"  ✓ Section '{section}' present")
            else:
                print(f"  ✗ Section '{section}' MISSING!")
                all_ok = False
        
        # Check drone categories
        drone_cats = config_data.get('drone_augmentation', {}).get('categories', [])
        expected_cats = ['drone_700m', 'drone_600m', 'drone_500m', 'drone_350m', 'drone_200m', 'drone_100m']
        
        print(f"\n  Drone categories found: {len(drone_cats)}")
        for cat in drone_cats:
            cat_name = cat.get('name', 'unknown')
            has_num_bg = 'num_background_noises' in cat
            status = "✓" if has_num_bg else "✗"
            print(f"    {status} {cat_name} (num_background_noises: {has_num_bg})")
        
        # Check ambient categories
        ambient_cats = config_data.get('no_drone_augmentation', {}).get('categories', [])
        print(f"\n  Ambient categories found: {len(ambient_cats)}")
        for cat in ambient_cats:
            cat_name = cat.get('name', 'unknown')
            has_amp = 'amplitude_range' in cat
            has_num = 'num_noise_sources' in cat
            status = "✓" if (has_amp and has_num) else "✗"
            print(f"    {status} {cat_name} (amplitude_range: {has_amp}, num_noise_sources: {has_num})")
        
        if all_ok:
            print("\n✅ Config v3 structure valid!")
        else:
            print("\n❌ Config v3 structure incomplete!")
        
        return all_ok
        
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON parse error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error reading config: {e}")
        return False


def test_convert_results_script():
    """Test that convert_results_for_viz includes Attention-CRNN."""
    print("\n" + "=" * 70)
    print("TEST 4: convert_results_for_viz.py Validation")
    print("=" * 70)
    
    script_path = Path("3 - Single Model Performance Calculation/convert_results_for_viz.py")
    
    if not script_path.exists():
        print(f"  ✗ Script not found: {script_path}")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check if Attention-CRNN is included
    has_attention_import = 'ATTENTION_CRNN_SCORES_PATH' in content
    has_attention_conversion = 'Attention-CRNN' in content
    has_4_models = '/4 models' in content or '4/4' in content
    
    print(f"  {'✓' if has_attention_import else '✗'} Imports ATTENTION_CRNN_SCORES_PATH")
    print(f"  {'✓' if has_attention_conversion else '✗'} Converts Attention-CRNN model")
    print(f"  {'✓' if has_4_models else '✗'} Mentions 4 models (not 3)")
    
    all_ok = has_attention_import and has_attention_conversion and has_4_models
    
    if all_ok:
        print("\n✅ convert_results_for_viz.py includes Attention-CRNN!")
    else:
        print("\n❌ convert_results_for_viz.py missing Attention-CRNN support!")
    
    return all_ok


def test_model_performance_script():
    """Test that model_performance.py includes Attention-CRNN."""
    print("\n" + "=" * 70)
    print("TEST 5: model_performance.py Validation")
    print("=" * 70)
    
    script_path = Path("6 - Visualization/model_performance.py")
    
    if not script_path.exists():
        print(f"  ✗ Script not found: {script_path}")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Count occurrences of model list with Attention-CRNN
    count_4_models = content.count("['CNN', 'RNN', 'CRNN', 'Attention-CRNN']")
    count_3_models = content.count("['CNN', 'RNN', 'CRNN']")
    
    print(f"  Found {count_4_models} lists with 4 models (includes Attention-CRNN)")
    print(f"  Found {count_3_models} lists with only 3 models (needs update)")
    
    all_ok = count_4_models > 0 and count_3_models == 0
    
    if all_ok:
        print("\n✅ model_performance.py fully supports Attention-CRNN!")
    else:
        print("\n⚠️  model_performance.py may have some 3-model lists remaining")
    
    return all_ok


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPATIBILITY TEST SUITE")
    print("=" * 70)
    
    results = {
        'Config Paths': test_config_paths(),
        'Visualization Imports': test_visualization_imports(),
        'Config v3 Structure': test_config_v3_structure(),
        'convert_results_for_viz': test_convert_results_script(),
        'model_performance.py': test_model_performance_script(),
    }
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("\n" + "=" * 70)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 70)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
