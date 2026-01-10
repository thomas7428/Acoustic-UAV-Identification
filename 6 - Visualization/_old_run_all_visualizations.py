"""
Complete Visualization Runner - Modern Version
Génère toutes les visualisations à partir des résultats JSON précalculés.

Usage:
    python run_all_visualizations.py
    python run_all_visualizations.py --skip-audio  # Skip audio examples generation
"""

import sys
from pathlib import Path
import argparse

# Import project config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from tools import plot_utils

# Import modern visualization modules
import modern_dataset_analysis
import modern_audio_examples
import modern_threshold_calibration
import performance_comparison

plot_utils.set_style()


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"{text:^80}")
    print("="*80 + "\n")


def main():
    """Run all modern visualization scripts."""
    parser = argparse.ArgumentParser(description='Run all modern visualization steps')
    parser.add_argument('--skip-audio', action='store_true', help='Skip audio examples generation')
    parser.add_argument('--skip-threshold', action='store_true', help='Skip threshold calibration')
    parser.add_argument('--best-only', action='store_true', default=True,
                        help='Run reduced visualizations using only best thresholds (default: True)')
    args = parser.parse_args()
    
    print_header("ACOUSTIC UAV IDENTIFICATION - MODERN VISUALIZATION SUITE")
    
    print("This script generates visualizations from precomputed JSON results.")
    print(f"Performance data source: {config.PERFORMANCE_DIR}")
    print(f"Outputs will be saved in: ./outputs/\n")
    
    success_count = 0
    total_steps = 4
    
    try:
        # 1. Performance Comparison (uses precomputed JSONs)
        print_header("STEP 1: PERFORMANCE COMPARISON")
        print("Loading results from JSON files and generating comparison plots...")
        try:
            # By default we run the reduced "best-only" comparison to avoid
            # generating many per-threshold PNGs. Use --best-only to enable.
            if args.best_only:
                import performance_comparison_best as pc_best
                pc_best.main()
            else:
                # Simuler l'appel avec --all
                sys.argv = [sys.argv[0], '--all']
                performance_comparison.main()

            success_count += 1
        except Exception as e:
            print(f"[WARNING] Performance comparison failed: {e}")
            print("Make sure to run Universal_Perf_Tester.py first to generate JSON files.")
        
        # 2. Dataset Analysis
        print_header("STEP 2: DATASET ANALYSIS")
        print("Analyzing dataset composition and statistics...")
        try:
            modern_dataset_analysis.main()
            success_count += 1
        except Exception as e:
            print(f"[WARNING] Dataset analysis failed: {e}")
        
        # 3. Audio Examples (optional)
        if not args.skip_audio:
            print_header("STEP 3: AUDIO EXAMPLES GENERATION")
            print("Generating representative audio examples with visualizations...")
            try:
                modern_audio_examples.main()
                success_count += 1
            except Exception as e:
                print(f"[WARNING] Audio examples generation failed: {e}")
        else:
            print_header("STEP 3: AUDIO EXAMPLES GENERATION [SKIPPED]")
            total_steps -= 1
        
        # 4. Threshold Calibration (optional) - skip when in best-only mode
        if args.best_only:
            print_header("STEP 4: THRESHOLD CALIBRATION [SKIPPED - best-only mode]")
            total_steps -= 1
        else:
            if not args.skip_threshold:
                print_header("STEP 4: THRESHOLD CALIBRATION")
                print("Analyzing threshold impact and generating recommendations...")
                try:
                    modern_threshold_calibration.main()
                    success_count += 1
                except Exception as e:
                    print(f"[WARNING] Threshold calibration failed: {e}")
                    print("Run Universal_Perf_Tester.py with multiple thresholds for this analysis.")
            else:
                print_header("STEP 4: THRESHOLD CALIBRATION [SKIPPED]")
                total_steps -= 1
        
        # Final summary
        print_header("VISUALIZATION SUITE COMPLETE")
        print(f"Successfully completed {success_count}/{total_steps} steps!")
        print(f"\nCheck the outputs in: {plot_utils.get_output_dir(__file__)}")
        print("\nGenerated files:")
        
        output_dir = Path(__file__).parent / 'outputs'
        if output_dir.exists():
            # Compter les fichiers par type
            png_files = list(output_dir.glob('*.png'))
            txt_files = list(output_dir.glob('*.txt'))
            json_files = list(output_dir.glob('*.json'))
            html_files = list(output_dir.glob('*.html'))
            
            print(f"  📊 {len(png_files)} PNG visualizations")
            print(f"  📄 {len(txt_files)} text reports")
            print(f"  🔧 {len(json_files)} JSON data files")
            print(f"  🌐 {len(html_files)} HTML pages")
            
            print(f"\n  Total: {len(list(output_dir.glob('*')))} files")
        
        print("\n" + "="*80)
        print("\n💡 Tips:")
        print("  - Use quick_viz.py for fast performance comparisons")
        print("  - Check performance_summary.txt for detailed metrics")
        print("  - Open audio_examples/index.html for audio samples")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n[ERROR] Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
