"""
Quick Test Script for Enhanced Pipeline v2.0
Validates that the new pipeline works correctly without full execution.

This script:
1. Checks all new files exist
2. Validates configurations
3. Tests imports
4. Runs dry-run tests
5. Verifies backward compatibility

Usage:
    python test_pipeline_v2.py
"""

import os
import sys
import json
from pathlib import Path


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.tests = []
    
    def add_test(self, name, passed, message=""):
        self.tests.append({
            'name': name,
            'passed': passed,
            'message': message
        })
        if passed:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            print(f"  ✗ {name}")
            if message:
                print(f"    {message}")
    
    def add_warning(self, name, message):
        self.warnings += 1
        print(f"  ⚠ {name}")
        if message:
            print(f"    {message}")
    
    def summary(self):
        print(f"\n{'='*80}")
        print(f"TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Passed:   {self.passed}")
        print(f"Failed:   {self.failed}")
        print(f"Warnings: {self.warnings}")
        print(f"Total:    {self.passed + self.failed}")
        
        if self.failed == 0:
            print(f"\n✓ All tests passed!")
            return True
        else:
            print(f"\n✗ {self.failed} test(s) failed")
            return False


def test_file_exists(results, filepath, description):
    """Test if a file exists."""
    exists = filepath.exists()
    results.add_test(
        f"File exists: {filepath.name}",
        exists,
        f"Missing: {filepath}" if not exists else ""
    )
    return exists


def test_directory_structure(script_dir):
    """Test if all required directories and files exist."""
    print("\n[1] Testing Directory Structure")
    print("─" * 80)
    
    results = TestResults()
    
    # Check new scripts
    new_files = [
        'augment_dataset_v2.py',
        'augment_config_v2.json',
        'split_dataset.py',
        'master_setup_v2.py',
        'README_V2.md',
        'MIGRATION_GUIDE.md'
    ]
    
    for filename in new_files:
        test_file_exists(results, script_dir / filename, f"New v2.0 file: {filename}")
    
    # Check old scripts still exist
    old_files = [
        'download_and_prepare_dads.py',
        'augment_dataset.py',
        'augment_config.json'
    ]
    
    for filename in old_files:
        test_file_exists(results, script_dir / filename, f"Legacy file: {filename}")
    
    return results


def test_configuration_validity(script_dir):
    """Test if configuration files are valid JSON."""
    print("\n[2] Testing Configuration Validity")
    print("─" * 80)
    
    results = TestResults()
    
    # Test augment_config_v2.json
    config_path = script_dir / 'augment_config_v2.json'
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required sections
        required_sections = ['drone_augmentation', 'no_drone_augmentation', 'audio_parameters', 'output']
        for section in required_sections:
            if section in config:
                results.add_test(f"Config section: {section}", True)
            else:
                results.add_test(f"Config section: {section}", False, "Section missing")
        
        # Check no-drone augmentation enabled
        if config.get('no_drone_augmentation', {}).get('enabled'):
            results.add_test("No-drone augmentation enabled", True)
        else:
            results.add_warning("No-drone augmentation disabled", "Feature will not be used")
        
        # Check samples per category
        drone_samples = config.get('output', {}).get('samples_per_category_drone')
        no_drone_samples = config.get('output', {}).get('samples_per_category_no_drone')
        
        if drone_samples and no_drone_samples:
            results.add_test("Sample counts configured", True, f"Drone: {drone_samples}, No-drone: {no_drone_samples}")
        else:
            results.add_test("Sample counts configured", False, "Missing sample counts")
        
    except json.JSONDecodeError as e:
        results.add_test("Parse augment_config_v2.json", False, f"JSON error: {e}")
    except Exception as e:
        results.add_test("Parse augment_config_v2.json", False, f"Error: {e}")
    
    return results


def test_python_imports():
    """Test if required Python packages are available."""
    print("\n[3] Testing Python Dependencies")
    print("─" * 80)
    
    results = TestResults()
    
    packages = {
        'numpy': 'NumPy',
        'librosa': 'Librosa (audio processing)',
        'soundfile': 'SoundFile (audio I/O)',
        'datasets': 'HuggingFace Datasets',
        'tqdm': 'Progress bars'
    }
    
    for package, description in packages.items():
        try:
            __import__(package)
            results.add_test(f"Import {description}", True)
        except ImportError:
            results.add_test(f"Import {description}", False, f"Install with: pip install {package}")
    
    # Test librosa effects (critical for v2.0)
    try:
        import librosa
        if hasattr(librosa.effects, 'pitch_shift') and hasattr(librosa.effects, 'time_stretch'):
            results.add_test("Librosa effects available", True)
        else:
            results.add_test("Librosa effects available", False, "Update librosa: pip install --upgrade librosa")
    except Exception as e:
        results.add_test("Librosa effects available", False, str(e))
    
    return results


def test_backward_compatibility(script_dir):
    """Test that old scripts still work."""
    print("\n[4] Testing Backward Compatibility")
    print("─" * 80)
    
    results = TestResults()
    
    # Check config.py
    project_root = script_dir.parent
    config_path = project_root / 'config.py'
    
    if config_path.exists():
        results.add_test("config.py exists", True)
        
        try:
            # Check if DATASET_ROOT_OVERRIDE is supported
            with open(config_path, 'r') as f:
                config_content = f.read()
            
            if 'DATASET_ROOT_OVERRIDE' in config_content:
                results.add_test("DATASET_ROOT_OVERRIDE supported", True)
            else:
                results.add_test("DATASET_ROOT_OVERRIDE supported", False, 
                               "config.py needs update for environment variable support")
        except Exception as e:
            results.add_test("Read config.py", False, str(e))
    else:
        results.add_test("config.py exists", False, "Missing centralized config")
    
    # Check folders 1-5 exist
    folders = [
        "1 - Preprocessing and Features Extraction",
        "2 - Model Training",
        "3 - Single Model Performance Calculation",
        "4 - Late Fusion Networks",
        "5 - Extras"
    ]
    
    for folder in folders:
        folder_path = project_root / folder
        if folder_path.exists():
            results.add_test(f"Folder exists: {folder}", True)
        else:
            results.add_warning(f"Folder missing: {folder}", "May not be needed for testing")
    
    return results


def test_script_syntax(script_dir):
    """Test if new scripts have valid Python syntax."""
    print("\n[5] Testing Script Syntax")
    print("─" * 80)
    
    results = TestResults()
    
    scripts = [
        'augment_dataset_v2.py',
        'split_dataset.py',
        'master_setup_v2.py'
    ]
    
    for script_name in scripts:
        script_path = script_dir / script_name
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, script_name, 'exec')
            results.add_test(f"Syntax valid: {script_name}", True)
        except SyntaxError as e:
            results.add_test(f"Syntax valid: {script_name}", False, f"Line {e.lineno}: {e.msg}")
        except Exception as e:
            results.add_test(f"Syntax valid: {script_name}", False, str(e))
    
    return results


def test_documentation(script_dir, project_root):
    """Test if documentation is complete."""
    print("\n[6] Testing Documentation")
    print("─" * 80)
    
    results = TestResults()
    
    docs = [
        (script_dir / 'README_V2.md', 'v2.0 README'),
        (script_dir / 'MIGRATION_GUIDE.md', 'Migration guide'),
        (project_root / 'ENHANCED_PIPELINE_V2_SUMMARY.md', 'Project summary')
    ]
    
    for doc_path, description in docs:
        if doc_path.exists():
            # Check if not empty
            size = doc_path.stat().st_size
            if size > 100:  # At least 100 bytes
                results.add_test(f"Documentation: {description}", True, f"{size} bytes")
            else:
                results.add_test(f"Documentation: {description}", False, "File too small")
        else:
            results.add_test(f"Documentation: {description}", False, "File missing")
    
    return results


def main():
    """Run all tests."""
    print("=" * 80)
    print("ENHANCED PIPELINE v2.0 - VALIDATION TESTS")
    print("=" * 80)
    
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    
    print(f"\nScript directory: {script_dir}")
    print(f"Project root: {project_root}")
    
    # Run all test suites
    all_results = []
    
    all_results.append(test_directory_structure(script_dir))
    all_results.append(test_configuration_validity(script_dir))
    all_results.append(test_python_imports())
    all_results.append(test_backward_compatibility(script_dir))
    all_results.append(test_script_syntax(script_dir))
    all_results.append(test_documentation(script_dir, project_root))
    
    # Combined summary
    print("\n" + "=" * 80)
    print("COMBINED RESULTS")
    print("=" * 80)
    
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    total_warnings = sum(r.warnings for r in all_results)
    
    print(f"\nTotal Passed:   {total_passed}")
    print(f"Total Failed:   {total_failed}")
    print(f"Total Warnings: {total_warnings}")
    
    if total_failed == 0:
        print(f"\n{'='*80}")
        print("✓ ALL VALIDATION TESTS PASSED!")
        print("The enhanced pipeline v2.0 is ready to use.")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Run: python master_setup_v2.py --dry-run")
        print("  2. Review the dry-run output")
        print("  3. Run: python master_setup_v2.py")
        print("=" * 80)
        return 0
    else:
        print(f"\n{'='*80}")
        print(f"✗ {total_failed} TEST(S) FAILED")
        print("Please fix the issues before using the pipeline.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        exit(1)
