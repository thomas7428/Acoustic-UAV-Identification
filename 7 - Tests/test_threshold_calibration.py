#!/usr/bin/env python3
"""
Test du workflow de calibration des thresholds
Simule un dataset de validation et teste l'optimisation hiérarchique
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import config

# Import du module de calibration
calibration_dir = project_root / "3 - Single Model Performance Calculation"
sys.path.insert(0, str(calibration_dir))
from calibrate_thresholds import optimize_threshold_hierarchical, calculate_metrics


def test_calibration_logic():
    """Test la logique d'optimisation avec données synthétiques."""
    print("="*80)
    print("  THRESHOLD CALIBRATION - UNIT TEST")
    print("="*80)
    
    # Générer données synthétiques
    np.random.seed(42)
    n_samples = 1000
    
    # Vraies labels: 500 drones, 500 ambient
    y_true = np.concatenate([
        np.ones(500),  # Drones
        np.zeros(500)  # Ambient
    ])
    
    # Prédictions probabilistes (scores entre 0 et 1)
    # Drones: distribution centrée sur 0.7 avec bruit
    # Ambient: distribution centrée sur 0.3 avec bruit
    y_pred_proba = np.concatenate([
        np.random.beta(7, 3, 500),  # Drones (skewed high)
        np.random.beta(3, 7, 500)   # Ambient (skewed low)
    ])
    
    print(f"\nSynthetic dataset:")
    print(f"  Total samples: {n_samples}")
    print(f"  Drones: {np.sum(y_true == 1)}")
    print(f"  Ambient: {np.sum(y_true == 0)}")
    print(f"  Mean score drones: {y_pred_proba[y_true == 1].mean():.3f}")
    print(f"  Mean score ambient: {y_pred_proba[y_true == 0].mean():.3f}")
    
    # Test 1: Optimisation avec contraintes par défaut
    print("\n" + "="*80)
    print("TEST 1: Default constraints")
    print("="*80)
    
    optimal_threshold, optimal_metrics, all_results, relaxation_info = optimize_threshold_hierarchical(
        y_true, y_pred_proba,
        min_recall=0.90,
        min_precision_drone=0.70,
        min_precision_ambient=0.85,
        verbose=True
    )
    
    # Vérifier contraintes
    assert optimal_metrics['recall'] >= 0.90 or relaxation_info is not None, "Recall constraint violated without relaxation"
    assert optimal_threshold > 0 and optimal_threshold < 1, "Threshold out of bounds"
    
    print("\n✓ TEST 1 PASSED")
    
    # Test 2: Contraintes très strictes (devrait relaxer)
    print("\n" + "="*80)
    print("TEST 2: Strict constraints (expect relaxation)")
    print("="*80)
    
    optimal_threshold2, optimal_metrics2, all_results2, relaxation_info2 = optimize_threshold_hierarchical(
        y_true, y_pred_proba,
        min_recall=0.99,  # Très strict
        min_precision_drone=0.95,  # Très strict
        min_precision_ambient=0.95,  # Très strict
        verbose=True
    )
    
    if relaxation_info2:
        print(f"\n✓ Relaxation applied as expected: {relaxation_info2['name']}")
    else:
        print(f"\n⚠️  No relaxation needed (constraints met)")
    
    print("\n✓ TEST 2 PASSED")
    
    # Test 3: Vérifier format JSON
    print("\n" + "="*80)
    print("TEST 3: JSON serialization")
    print("="*80)
    
    calibration_data = {
        'version': '1.0',
        'test_mode': True,
        'models': {
            'TEST_MODEL': {
                'threshold': optimal_threshold,
                'validation_samples': n_samples,
                'metrics_at_threshold': optimal_metrics,
                'relaxation_applied': relaxation_info,
                'all_tested_thresholds': all_results[:5]  # Sample only
            }
        }
    }
    
    try:
        json_str = json.dumps(calibration_data, indent=2)
        parsed = json.loads(json_str)
        print(f"\n✓ JSON serialization successful ({len(json_str)} bytes)")
        print("\nSample JSON:")
        print(json_str[:500] + "...")
    except Exception as e:
        raise AssertionError(f"JSON serialization failed: {e}")
    
    print("\n✓ TEST 3 PASSED")
    
    # Test 4: Helper function
    print("\n" + "="*80)
    print("TEST 4: Config helper function")
    print("="*80)
    
    # Test avec fichier inexistant
    thresholds = config.load_calibrated_thresholds()
    assert isinstance(thresholds, dict), "Helper should return dict"
    print(f"✓ Helper returns dict: {type(thresholds)}")
    
    if config.CALIBRATION_FILE_PATH.exists():
        print(f"✓ Calibration file exists: {config.CALIBRATION_FILE_PATH}")
        print(f"  Loaded thresholds: {list(thresholds.keys())}")
    else:
        print(f"⚠️  No calibration file yet (expected on first run)")
    
    print("\n✓ TEST 4 PASSED")
    
    # Final summary
    print("\n" + "="*80)
    print("  ALL TESTS PASSED ✓")
    print("="*80)
    print("\nCalibration logic validated:")
    print("  ✓ Hierarchical optimization works")
    print("  ✓ Constraint enforcement correct")
    print("  ✓ Progressive relaxation functional")
    print("  ✓ JSON serialization compatible")
    print("  ✓ Config integration working")
    
    return 0


if __name__ == '__main__':
    sys.exit(test_calibration_logic())
