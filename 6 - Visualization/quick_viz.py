#!/usr/bin/env python3
"""
Quick Performance Visualization Launcher
Lance rapidement des visualisations courantes avec des présets.
"""

import subprocess
import sys
from pathlib import Path

RUNNER_SCRIPT = Path(__file__).parent / "run_visualizations.py"
PERF_SCRIPT = Path(__file__).parent / "performance_comparison_best.py"

PRESETS = {
    "all": {
        "description": "Pipeline complet de visualisations",
        "script": RUNNER_SCRIPT,
        "args": []
    },
    "fast": {
        "description": "Pipeline rapide (sans audio examples ni threshold calibration)",
        "script": RUNNER_SCRIPT,
        "args": ["--skip-audio", "--skip-threshold"]
    },
    "performance": {
        "description": "Comparaison de performances uniquement (meilleurs thresholds)",
        "script": PERF_SCRIPT,
        "args": []
    },
    "no-audio": {
        "description": "Pipeline complet sans génération d'exemples audio",
        "script": RUNNER_SCRIPT,
        "args": ["--skip-audio"]
    }
}

def print_presets():
    """Affiche les presets disponibles."""
    print("\n" + "="*80)
    print("  QUICK VISUALIZATION PRESETS")
    print("="*80)
    for name, info in PRESETS.items():
        print(f"\n  {name:15s} - {info['description']}")
    print("\n" + "="*80)
    print("\nUsage: python quick_viz.py <preset_name>")
    print("Example: python quick_viz.py all")
    print("         python quick_viz.py fast\n")

def main():
    if len(sys.argv) < 2:
        print_presets()
        sys.exit(0)
    
    preset_name = sys.argv[1]
    
    if preset_name == "help" or preset_name == "-h" or preset_name == "--help":
        print_presets()
        sys.exit(0)
    
    if preset_name in PRESETS:
        # Mode preset
        preset = PRESETS[preset_name]
        script = preset["script"]
        args = preset["args"]
        
        print(f"\n🚀 Lancement du preset: {preset_name}")
        print(f"   {preset['description']}")
        print(f"   Script: {script.name}\n")
    else:
        print(f"\n❌ Preset '{preset_name}' inconnu.")
        print_presets()
        sys.exit(1)
    
    # Lancer le script
    cmd = [sys.executable, str(script)] + args
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
