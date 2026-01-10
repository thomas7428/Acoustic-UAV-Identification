#!/usr/bin/env python3
"""
Enhanced Visualization Runner
Exécute toutes les nouvelles visualisations dans le bon ordre.
"""
import subprocess
import sys
from pathlib import Path
import json


# Import config for debug info
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def run_script(script_name, description=""):
    """Run a visualization script and report status."""
    print(f"\n{'='*80}")
    print(f"  {description or script_name}")
    print(f"{'='*80}\n")
    
    # Parse script name and arguments
    parts = script_name.split()
    script_file = parts[0]
    script_args = parts[1:] if len(parts) > 1 else []
    
    script_path = Path(__file__).parent / script_file
    
    if not script_path.exists():
        print(f"[WARNING] Script not found: {script_path} (skipping)")
        return None
    
    try:
        cmd = [sys.executable, str(script_path)] + script_args
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("[STDERR]:", result.stderr)
        print(f"✓ Completed: {script_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {script_file} (exit code {e.returncode})")
        print("[STDOUT]:", e.stdout)
        print("[STDERR]:", e.stderr)
        return False


def main():
    print("=" * 80)
    print("  ENHANCED VISUALIZATION PIPELINE")
    print("=" * 80)
    print("\nThis script runs all new visualization tools in the correct order.")
    print()
    # Debug: show which calibration file and thresholds mapping will be used by visualizers
    try:
        print(f"[DEBUG] config.CALIBRATION_FILE_PATH: {getattr(config, 'CALIBRATION_FILE_PATH', None)}")
        print(f"[DEBUG] config.CALIBRATION_FILE_PATH (str): {str(getattr(config, 'CALIBRATION_FILE_PATH', ''))}")
        print(f"[DEBUG] MODEL_THRESHOLDS_NORMALIZED: {json.dumps(getattr(config, 'MODEL_THRESHOLDS_NORMALIZED', {}), indent=2)}")
    except Exception as e:
        print(f"[DEBUG] Failed to print config thresholds: {e}")
    
    # Ordered execution pipeline
    pipeline = [
        ("performance_comparison_best.py", "Step 1: Performance Analysis (Best Thresholds Only)"),
        ("threshold_calibration_comparison.py", "Step 3: Threshold Calibration Analysis"),
        ("model_comparison_plots.py", "Step 4: Model Performance Comparison"),
        ("snr_distribution.py", "Step 5: SNR Distribution Analysis"),
        ("modern_dataset_analysis.py", "Step 6: Dataset Analysis"),
        ("modern_threshold_calibration.py", "Step 7: Modern Threshold Calibration"),
        ("generate_html_report.py", "Step 8: Generate HTML Report"),
    ]
    
    results = {}
    
    for script, description in pipeline:
        result = run_script(script, description)
        if result is not None:  # None = script not found (optional)
            # Extract script name without arguments for results dict
            script_name = script.split()[0]
            results[script_name] = result
    
    # Summary
    print(f"\n{'='*80}")
    print("  EXECUTION SUMMARY")
    print(f"{'='*80}")
    
    for script, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{script:<45} {status}")
    
    total = len(results)
    successful = sum(results.values())
    failed = total - successful
    
    print(f"\n{'='*80}")
    print(f"  {successful}/{total} scripts completed successfully")
    
    if failed > 0:
        print(f"  {failed} script(s) failed")
        print(f"{'='*80}")
        sys.exit(1)
    
    print(f"{'='*80}")
    print("\n✓ All visualizations completed successfully!")
    
    # Show output location
    viz_dir = Path(__file__).parent / "outputs"
    print(f"\nResults location: {viz_dir}")
    
    if viz_dir.exists():
        files = list(viz_dir.glob("*.png")) + list(viz_dir.glob("*.html"))
        if files:
            print(f"\nGenerated files ({len(files)}):")
            for f in sorted(files):
                print(f"  - {f.name}")


if __name__ == '__main__':
    main()
