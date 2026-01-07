#!/usr/bin/env python3
"""
Obsolete: `select_best_results.py` maintained for history but not used.
Visualizations now read canonical performance files directly from
`config.PERFORMANCE_DIR` with filenames `{model_lower}_{split}.json`.
"""
import sys
print("select_best_results.py is obsolete. Visualizations load canonical files directly.")
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Debug: print calibration file path and thresholds mapping
try:
    print(f"[DEBUG select_best_results] CALIBRATION_FILE_PATH={getattr(config,'CALIBRATION_FILE_PATH',None)}")
    print(f"[DEBUG select_best_results] MODEL_THRESHOLDS_NORMALIZED={config.MODEL_THRESHOLDS_NORMALIZED}")
except Exception:
    # Minimal entrypoint: strictly use calibration file and corresponding perf JSONs
    print("=" * 80)
    print("  CALIBRATION-DRIVEN RESULTS SUMMARY")
    print("=" * 80)

    results = load_and_select_best_results()
    if not results:
        print("[ERROR] Could not load calibrated performance results.")
        return

    # Save a simple summary JSON (overwrite existing summary)
    output_dir = Path(config.PERFORMANCE_DIR)
    output_file = output_dir / "best_results_summary.json"
    save_best_results_summary(results, output_file)

    # Print a compact table based strictly on calibrated thresholds
    print("\nSUMMARY (calibration-driven)")
    print(f"{ 'Model':<15} {'Split':<8} {'Threshold':<10} {'F1':<8} {'Acc':<8} {'Prec':<8} {'Recall':<8}")
    print("-" * 80)
    for (model, split), data in sorted(results.items()):
        metrics = data.get('global_metrics', {})
        metadata = data.get('metadata', {})
        threshold = metadata.get('threshold', 'NA')
        f1 = metrics.get('f1_score', 0)
        acc = metrics.get('accuracy', 0)
        prec = metrics.get('precision', 0)
        rec = metrics.get('recall', 0)
        print(f"{model:<15} {split:<8} {float(threshold):<10.3f} {f1:<8.4f} {acc:<8.4f} {prec:<8.4f} {rec:<8.4f}")

    print("=" * 80)
    return results
    for model_key, threshold in normalized_calib.items():
        # filename model token expected in the perf filenames
        model_token = model_key.lower()
        for split in ('train', 'val', 'test'):
            pattern = f"{model_token}_{split}_t{float(threshold):.2f}_*.json"
            matches = list(perf_dir.glob(pattern))
            if not matches:
                print(f"[ERROR] No performance file found for {model_key} {split} at threshold {threshold:.2f} (expected pattern: {pattern})")
                return {}
            # If multiple matches, choose the most recent by name (timestamp in filename)
            matches.sort()
            chosen = matches[-1]
            try:
                with open(chosen, 'r') as f:
                    data = json.load(f)
                best_results[(model_key, split)] = data
                print(f"  ✓ Loaded calibrated result for {model_key} {split}: {chosen.name}")
            except Exception as e:
                print(f"  ✗ Error loading {chosen.name}: {e}")
                return {}

    return best_results


def save_best_results_summary(best_results, output_file):
    """
    Sauvegarde un résumé des meilleurs résultats.
    """
    summary = {
        'selection_criteria': 'Best F1-Score per (model, split)',
        'total_configurations': len(best_results),
        'results': {}
    }
    
    for (model, split), data in best_results.items():
        key = f"{model}_{split}"
        
        metrics = data.get('global_metrics', {})
        metadata = data.get('metadata', {})
        
        summary['results'][key] = {
            'model': model,
            'split': split,
            'threshold': metadata.get('threshold', 0.5),
            'metrics': {
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'specificity': metrics.get('specificity', 0)
            },
            'confusion_matrix': {
                'tp': metrics.get('tp', 0),
                'tn': metrics.get('tn', 0),
                'fp': metrics.get('fp', 0),
                'fn': metrics.get('fn', 0)
            }
        }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved: {output_file}")


def main():
    print("=" * 80)
    print("  BEST RESULTS SELECTOR")
    print("=" * 80)
    print()
    
    # Charger et sélectionner les meilleurs
    best_results = load_and_select_best_results()
    
    if not best_results:
        print("[ERROR] No results found!")
        return
    
    print(f"\n[INFO] Selected {len(best_results)} best configurations")
    
    # Sauvegarder le résumé
    output_dir = Path(config.PERFORMANCE_DIR)
    output_file = output_dir / "best_results_summary.json"
    save_best_results_summary(best_results, output_file)
    
    # Afficher tableau récapitulatif
    print("\n" + "=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model':<15} {'Split':<8} {'Threshold':<10} {'F1':<8} {'Acc':<8} {'Prec':<8} {'Recall':<8}")
    print("-" * 80)
    
    for (model, split), data in sorted(best_results.items()):
        metrics = data.get('global_metrics', {})
        metadata = data.get('metadata', {})
        
        threshold = metadata.get('threshold', 0.5)
        f1 = metrics.get('f1_score', 0)
        acc = metrics.get('accuracy', 0)
        prec = metrics.get('precision', 0)
        rec = metrics.get('recall', 0)
        
        print(f"{model:<15} {split:<8} {threshold:<10.3f} {f1:<8.4f} {acc:<8.4f} {prec:<8.4f} {rec:<8.4f}")
    
    print("=" * 80)
    
    return best_results


if __name__ == '__main__':
    main()
