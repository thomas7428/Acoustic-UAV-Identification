"""
Complete Visualization Runner
Generates all visualizations and creates a comprehensive analysis report.

Usage:
    python run_all_visualizations.py
"""

import sys
from pathlib import Path

# Import visualization modules
import argparse
import dataset_analysis
import model_performance
import augmentation_impact
import performance_by_distance


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"{text:^80}")
    print("="*80 + "\n")


def main():
    """Run all visualization scripts."""
    parser = argparse.ArgumentParser(description='Run all visualization steps')
    parser.add_argument('--force-retest', action='store_true', help='Force re-running distance performance even if cached results exist')
    args = parser.parse_args()
    print_header("ACOUSTIC UAV IDENTIFICATION - VISUALIZATION SUITE")
    
    print("This script will generate all visualizations and analysis reports.")
    print("Outputs will be saved in: ./outputs/\n")
    
    try:
        # 1. Performance by distance (produce canonical artifacts used by other visualizations)
        print_header("STEP 1: PERFORMANCE BY DISTANCE (GENERATE CANONICAL ARTIFACTS)")
        performance_by_distance.main(force_retest=args.force_retest)

        # 2. Dataset Analysis
        print_header("STEP 2: DATASET ANALYSIS")
        dataset_analysis.main()

        # 3. Model Performance
        print_header("STEP 3: MODEL PERFORMANCE ANALYSIS")
        model_performance.main()
        
        # 3. Augmentation Impact
        print_header("STEP 3: AUGMENTATION IMPACT ANALYSIS")
        augmentation_impact.main()
        
        # NOTE: `performance_by_distance` already ran at STEP 1 to produce canonical artifacts.
        # We avoid running it a second time here to prevent duplicate long computations.
        
        # Final summary
        print_header("VISUALIZATION SUITE COMPLETE")
        print("All visualizations have been generated successfully!")
        print(f"\nCheck the outputs in: {Path(__file__).parent / 'outputs'}")
        print("\nGenerated files:")
        
        output_dir = Path(__file__).parent / 'outputs'
        if output_dir.exists():
            files = sorted(output_dir.glob('*'))
            for i, file in enumerate(files, 1):
                print(f"  {i}. {file.name}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n[ERROR] Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
