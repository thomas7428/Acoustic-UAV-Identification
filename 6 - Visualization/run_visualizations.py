#!/usr/bin/env python3
"""
Unified Visualization Pipeline
Génère toutes les visualisations à partir des résultats JSON précalculés.

Usage:
    python run_visualizations.py                    # Pipeline complet
    python run_visualizations.py --skip-audio       # Sans audio examples
    python run_visualizations.py --skip-threshold   # Sans threshold calibration
"""

import sys
from pathlib import Path
import argparse

# Import project config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from tools import plot_utils

# Import visualization modules
import select_best_results
import performance_comparison_best
import threshold_calibration_comparison
import model_comparison_plots
import snr_distribution
import modern_dataset_analysis
import modern_threshold_calibration
import modern_audio_examples
import threshold_analysis
import generate_html_report

plot_utils.set_style()


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"{text:^80}")
    print("="*80 + "\n")


def print_step(step_num, total, name):
    """Print step header."""
    print(f"\n{'='*80}")
    print(f"  [{step_num}/{total}] {name}")
    print(f"{'='*80}\n")


def main():
    """Run unified visualization pipeline."""
    parser = argparse.ArgumentParser(description='Run complete visualization pipeline')
    parser.add_argument('--skip-audio', action='store_true', 
                        help='Skip audio examples generation (slower)')
    parser.add_argument('--skip-threshold', action='store_true',
                        help='Skip threshold calibration analysis')
    args = parser.parse_args()
    
    print_header("ACOUSTIC UAV IDENTIFICATION - VISUALIZATION PIPELINE")
    
    print("This script generates visualizations from precomputed JSON results.")
    print(f"Performance data source: {config.PERFORMANCE_DIR}")
    print(f"Outputs will be saved in: ./outputs/\n")
    
    # Define pipeline steps
    total_steps = 8
    if args.skip_audio:
        total_steps -= 1
    if args.skip_threshold:
        total_steps -= 1
    
    steps = []
    step_num = 1
    
    # Step 0: Select best results (prerequisite)
    steps.append((step_num, "Select Best Results", select_best_results.main,
                  "Analyzing all performance JSONs to identify best configurations..."))
    step_num += 1
    
    # Step 1: Performance comparison
    steps.append((step_num, "Performance Comparison (Best Thresholds)", performance_comparison_best.main,
                  "Loading best results and generating comparison plots..."))
    step_num += 1
    
    # Step 2: Threshold calibration comparison
    steps.append((step_num, "Threshold Calibration Comparison", threshold_calibration_comparison.main,
                  "Analyzing threshold impact across models..."))
    step_num += 1
    
    # Step 3: Model comparison
    steps.append((step_num, "Model Performance Comparison", model_comparison_plots.main,
                  "Generating model comparison visualizations..."))
    step_num += 1
    
    # Step 4: SNR distribution
    steps.append((step_num, "SNR Distribution Analysis", snr_distribution.main,
                  "Analyzing Signal-to-Noise Ratio distribution..."))
    step_num += 1
    
    # Step 5: Dataset analysis
    steps.append((step_num, "Dataset Composition Analysis", modern_dataset_analysis.main,
                  "Analyzing dataset statistics and distributions..."))
    step_num += 1
    
    # Step 6: Threshold calibration analysis
    if not args.skip_threshold and config.CALIBRATION_FILE_PATH.exists():
        steps.append((step_num, "Threshold Calibration Analysis", threshold_analysis.main,
                      "Analyzing calibrated thresholds and optimization curves..."))
        step_num += 1
    
    # Step 7: Legacy threshold calibration (optional)
    if not args.skip_threshold:
        steps.append((step_num, "Legacy Threshold Calibration", modern_threshold_calibration.main,
                      "Generating threshold recommendations..."))
        step_num += 1
    
    # Step 8: Audio examples (optional)
    if not args.skip_audio:
        steps.append((step_num, "Audio Examples Generation", modern_audio_examples.main,
                      "Generating representative audio examples with visualizations..."))
        step_num += 1
    
    # Step 9: HTML report
    steps.append((step_num, "HTML Report Generation", generate_html_report.main,
                  "Creating interactive HTML report..."))
    
    # Execute pipeline
    success_count = 0
    failed_steps = []
    
    for num, name, func, description in steps:
        print_step(num, total_steps, name)
        print(description)
        try:
            func()
            success_count += 1
            print(f"\n✓ {name} completed successfully")
        except Exception as e:
            failed_steps.append(name)
            print(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n[WARNING] Continuing with remaining steps...")
    
    # Final summary
    print_header("VISUALIZATION PIPELINE COMPLETE")
    print(f"Successfully completed {success_count}/{total_steps} steps")
    
    if failed_steps:
        print(f"\n⚠️  Failed steps ({len(failed_steps)}):")
        for step in failed_steps:
            print(f"  - {step}")
    
    output_dir = Path(__file__).parent / 'outputs'
    if output_dir.exists():
        # Count generated files
        png_files = list(output_dir.glob('*.png'))
        txt_files = list(output_dir.glob('*.txt'))
        json_files = list(output_dir.glob('*.json'))
        html_files = list(output_dir.glob('*.html'))
        
        print(f"\nGenerated outputs in: {output_dir}")
        print(f"  📊 {len(png_files)} PNG visualizations")
        print(f"  📄 {len(txt_files)} text reports")
        print(f"  🔧 {len(json_files)} JSON data files")
        print(f"  🌐 {len(html_files)} HTML pages")
        print(f"\n  Total: {len(list(output_dir.glob('*')))} files")
    
    print("\n" + "="*80)
    print("\n💡 Quick Access:")
    print(f"  📊 Performance Report: {output_dir / 'performance_report.html'}")
    print(f"  📈 Best Results: {config.PERFORMANCE_DIR / 'best_results_summary.json'}")
    if not args.skip_audio and (output_dir / 'audio_examples' / 'index.html').exists():
        print(f"  🔊 Audio Examples: {output_dir / 'audio_examples' / 'index.html'}")
    print("\n" + "="*80)
    
    return 0 if success_count == total_steps else 1


if __name__ == "__main__":
    sys.exit(main())
