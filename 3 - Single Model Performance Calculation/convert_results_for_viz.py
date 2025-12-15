"""
Convert performance results from folder 3 format to visualization-compatible format.

This script converts the basic performance scores (TP, FN, TN, FP, Accuracy, etc.)
into the format expected by the visualization scripts (confusion matrix, classification report, etc.)
"""

import json
import sys
from pathlib import Path
import numpy as np

# Import centralized configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def convert_scores_to_accuracy_format(scores_path, output_path, model_name):
    """
    Convert basic scores format to visualization-compatible format.
    
    Args:
        scores_path: Path to the scores JSON file (e.g., cnn_scores.json)
        output_path: Path where to save the converted file (e.g., results/cnn_accuracy.json)
        model_name: Name of the model (for display purposes)
    """
    # Load scores
    with open(scores_path, 'r') as f:
        scores = json.load(f)
    
    # Support multiple score formats produced by different scripts:
    # 1) Old format: keys 'TP','FN','TN','FP','Accuracy','Precision','Recall','F1 Score' as single-item lists
    # 2) Newer/simple format: flat fields 'accuracy','precision','recall','f1_score' and a 'confusion_matrix' dict
    if all(k in scores for k in ('TP', 'FN', 'TN', 'FP', 'Accuracy', 'Precision', 'Recall', 'F1 Score')):
        TP = scores['TP'][0]
        FN = scores['FN'][0]
        TN = scores['TN'][0]
        FP = scores['FP'][0]
        accuracy = scores['Accuracy'][0]
        precision = scores['Precision'][0]
        recall = scores['Recall'][0]
        f1_score = scores['F1 Score'][0]
    elif 'confusion_matrix' in scores:
        # Expecting confusion_matrix to be a dict with TP,FN,TN,FP keys (as produced by newer performance scripts)
        cm = scores.get('confusion_matrix', {})
        TP = int(cm.get('TP') or cm.get('tp') or 0)
        FN = int(cm.get('FN') or cm.get('fn') or 0)
        TN = int(cm.get('TN') or cm.get('tn') or 0)
        FP = int(cm.get('FP') or cm.get('fp') or 0)
        # Fall back to either lowercase or capitalized names
        accuracy = float(scores.get('accuracy', scores.get('Accuracy', 0)))
        precision = float(scores.get('precision', scores.get('Precision', 0)))
        recall = float(scores.get('recall', scores.get('Recall', 0)))
        f1_score = float(scores.get('f1_score', scores.get('F1 Score', scores.get('f1', 0))))
    else:
        raise ValueError("Unknown scores format: cannot extract TP/FN/TN/FP or expected keys from scores file")
    
    # Build confusion matrix
    # Format: [[TN, FP], [FN, TP]]
    # Rows = True labels (0: No Drone, 1: Drone)
    # Cols = Predicted labels (0: No Drone, 1: Drone)
    confusion_matrix = [
        [int(TN), int(FP)],  # True label = 0 (No Drone)
        [int(FN), int(TP)]   # True label = 1 (Drone)
    ]
    
    # Build classification report (sklearn format)
    classification_report = {
        "0": {  # Class 0: No Drone
            "precision": TN / (TN + FN) if (TN + FN) > 0 else 0,
            "recall": TN / (TN + FP) if (TN + FP) > 0 else 0,
            "f1-score": 0,  # Calculated below
            "support": int(TN + FP)
        },
        "1": {  # Class 1: Drone
            "precision": precision,
            "recall": recall,
            "f1-score": f1_score,
            "support": int(TP + FN)
        },
        "accuracy": accuracy,
        "macro avg": {
            "precision": 0,  # Calculated below
            "recall": 0,     # Calculated below
            "f1-score": 0,   # Calculated below
            "support": int(TP + FN + TN + FP)
        },
        "weighted avg": {
            "precision": precision,
            "recall": recall,
            "f1-score": f1_score,
            "support": int(TP + FN + TN + FP)
        }
    }
    
    # Calculate missing metrics for class 0
    if classification_report["0"]["precision"] > 0 and classification_report["0"]["recall"] > 0:
        classification_report["0"]["f1-score"] = (
            2 * classification_report["0"]["precision"] * classification_report["0"]["recall"] /
            (classification_report["0"]["precision"] + classification_report["0"]["recall"])
        )
    
    # Calculate macro averages
    classification_report["macro avg"]["precision"] = (
        classification_report["0"]["precision"] + classification_report["1"]["precision"]
    ) / 2
    classification_report["macro avg"]["recall"] = (
        classification_report["0"]["recall"] + classification_report["1"]["recall"]
    ) / 2
    classification_report["macro avg"]["f1-score"] = (
        classification_report["0"]["f1-score"] + classification_report["1"]["f1-score"]
    ) / 2
    
    # Build final output format
    output_data = {
        "model_name": model_name,
        "test_accuracy": accuracy,
        "test_loss": None,  # Not available from scores
        "confusion_matrix": confusion_matrix,
        "classification_report": classification_report,
        "metrics": {
            "TP": int(TP),
            "FN": int(FN),
            "TN": int(TN),
            "FP": int(FP),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    }
    
    # Save converted format
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"[OK] Converted {model_name} scores to: {output_path}")
    print(f"     Accuracy: {accuracy*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1: {f1_score*100:.2f}%")


def main():
    """Convert all available model scores."""
    
    print("=" * 60)
    print("Converting Performance Results for Visualization")
    print("=" * 60)
    
    conversions = [
        (config.CNN_SCORES_PATH, config.CNN_ACC_PATH, "CNN"),
        (config.RNN_SCORES_PATH, config.RNN_ACC_PATH, "RNN"),
        (config.CRNN_SCORES_PATH, config.CRNN_ACC_PATH, "CRNN"),
        (config.ATTENTION_CRNN_SCORES_PATH, config.ATTENTION_CRNN_ACC_PATH, "Attention-CRNN")
    ]
    
    converted_count = 0
    
    for scores_path, output_path, model_name in conversions:
        if scores_path.exists():
            try:
                convert_scores_to_accuracy_format(scores_path, output_path, model_name)
                converted_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to convert {model_name}: {e}")
        else:
            print(f"[SKIP] {model_name} scores not found: {scores_path}")
    
    print("=" * 60)
    print(f"Conversion complete: {converted_count}/4 models converted")
    print("=" * 60)
    
    if converted_count > 0:
        print("\n✓ You can now run the visualization scripts!")
        print(f"  Results saved to: {config.RESULTS_DIR}")
    else:
        print("\n✗ No model scores found. Please run the performance calculation scripts first:")
        print("  - CNN_Performance_Calcs.py (for CNN)")
        print("  - RNN_Performance_Calcs.py (for RNN)")
        print("  - CRNN_Performance_Calcs.py (for CRNN)")
        print("  - Attention_CRNN_Performance_Calcs.py (for Attention-CRNN)")


if __name__ == "__main__":
    main()
