#!/usr/bin/env python3
"""
Quick Performance Visualization Launcher
Lance rapidement des visualisations courantes avec des présets.
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent / "performance_comparison.py"

PRESETS = {
    "all": {
        "description": "Visualiser TOUS les résultats disponibles",
        "args": ["--all"]
    },
    "cnn-test": {
        "description": "Performances CNN sur test set uniquement",
        "args": ["--models", "CNN", "--splits", "test"]
    },
    "compare-models": {
        "description": "Comparer tous les modèles sur test set avec threshold 0.5",
        "args": ["--splits", "test", "--thresholds", "0.5"]
    },
    "threshold-analysis": {
        "description": "Analyser l'impact des thresholds (0.4 à 0.7)",
        "args": ["--models", "CNN", "--splits", "test", "--thresholds", "0.4", "0.5", "0.6", "0.7"]
    },
    "train-val-test": {
        "description": "Comparer performances sur train, val et test",
        "args": ["--models", "CNN", "--splits", "train", "val", "test"]
    }
}

def print_presets():
    """Affiche les presets disponibles."""
    print("\n" + "="*80)
    print("PRESETS DISPONIBLES")
    print("="*80)
    for name, info in PRESETS.items():
        print(f"\n  {name:20s} - {info['description']}")
    print("\n" + "="*80)
    print("\nUsage: python quick_viz.py <preset_name>")
    print("   ou: python quick_viz.py custom --models CNN RNN --splits test\n")

def main():
    if len(sys.argv) < 2:
        print_presets()
        sys.exit(0)
    
    preset_name = sys.argv[1]
    
    if preset_name == "help" or preset_name == "-h" or preset_name == "--help":
        print_presets()
        sys.exit(0)
    
    if preset_name == "custom":
        # Mode custom : passer tous les arguments restants
        args = sys.argv[2:]
    elif preset_name in PRESETS:
        # Mode preset
        args = PRESETS[preset_name]["args"]
        print(f"\n🚀 Lancement du preset: {preset_name}")
        print(f"   {PRESETS[preset_name]['description']}\n")
    else:
        print(f"\n❌ Preset '{preset_name}' inconnu.")
        print_presets()
        sys.exit(1)
    
    # Lancer le script principal
    cmd = [sys.executable, str(SCRIPT_PATH)] + args
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
