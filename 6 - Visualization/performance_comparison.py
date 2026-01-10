#!/usr/bin/env python3
"""
DEPRECATED: This script is obsolete
Use performance_comparison_best.py instead

This legacy script (640 lines) generated too many PNGs and had complex argparse.
The new approach uses best_results_summary.json for cleaner visualizations.

Migration:
  Old: python performance_comparison.py --all
  New: python run_visualizations.py
  
  Old: python performance_comparison.py --models CNN --splits test
  New: python performance_comparison_best.py (loads best results automatically)
"""

import sys
from pathlib import Path

print("\n" + "="*80)
print("  ⚠️  DEPRECATED SCRIPT")
print("="*80)
print("\nThis script has been replaced by performance_comparison_best.py")
print("\nRecommended usage:")
print("  • Full pipeline:     python run_visualizations.py")
print("  • Performance only:  python performance_comparison_best.py")
print("  • Quick presets:     python quick_viz.py")
print("\n" + "="*80)
print("\nFor legacy functionality, use: _deprecated_performance_comparison.py")
print("="*80 + "\n")

sys.exit(1)
