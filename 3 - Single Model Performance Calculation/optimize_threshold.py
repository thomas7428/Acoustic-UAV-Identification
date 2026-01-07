#!/usr/bin/env python3
"""
Threshold Optimizer
Trouve le seuil optimal pour un modèle en testant plusieurs valeurs sur le dataset de validation.
Retourne le meilleur threshold basé sur le F1-score.
"""
import json
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def test_threshold(model, X, y, threshold):
    """Test un seuil et retourne les métriques."""
    predictions = model.predict(X, verbose=0)
    y_pred = (predictions[:, 1] > threshold).astype(int)
    
    tp = np.sum((y == 1) & (y_pred == 1))
    tn = np.sum((y == 0) & (y_pred == 0))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / len(y) if len(y) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def main():
    parser = argparse.ArgumentParser(description='Optimise le threshold sur le dataset de validation')
    parser.add_argument('--model', type=str, required=True,
                        choices=['CNN', 'RNN', 'CRNN', 'ATTENTION_CRNN'],
                        help='Modèle à optimiser')
    parser.add_argument('--min', type=float, default=0.3,
                        help='Seuil minimum à tester (défaut: 0.3)')
    parser.add_argument('--max', type=float, default=0.7,
                        help='Seuil maximum à tester (défaut: 0.7)')
    parser.add_argument('--step', type=float, default=0.05,
                        help='Pas entre chaque seuil (défaut: 0.05)')
    args = parser.parse_args()
    
    # Charger le modèle
    model_paths = {
        'CNN': config.CNN_MODEL_PATH,
        'RNN': config.RNN_MODEL_PATH,
        'CRNN': config.CRNN_MODEL_PATH,
        'ATTENTION_CRNN': config.ATTENTION_CRNN_MODEL_PATH
    }
    model_path = model_paths[args.model]
    print(f"Chargement du modèle {args.model} depuis {model_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)
    
    # Charger les features de validation
    mel_path = config.MEL_VAL_DATA_PATH
    print(f"Chargement des features depuis {mel_path}")
    
    with open(mel_path, 'r') as f:
        data = json.load(f)
    
    mels = np.array(data['mel'])
    labels = np.array(data.get('labels', []))
    
    if len(labels) != len(mels):
        raise RuntimeError(f"Labels manquants ou incompatibles dans {mel_path}")
    
    # Expand dimensions pour CNN/CRNN
    if args.model in ['CNN', 'CRNN', 'ATTENTION_CRNN']:
        mels = np.expand_dims(mels, axis=-1)
    
    print(f"\nOptimisation du threshold sur validation set ({len(mels)} échantillons)")
    print(f"Plage testée: [{args.min}, {args.max}] avec pas de {args.step}")
    print("=" * 80)
    
    # Tester différents thresholds
    thresholds = np.arange(args.min, args.max + args.step, args.step)
    results = []
    
    for threshold in thresholds:
        metrics = test_threshold(model, mels, labels, threshold)
        results.append(metrics)
        print(f"Threshold {threshold:.2f}: Acc={metrics['accuracy']:.4f}, "
              f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, "
              f"F1={metrics['f1_score']:.4f}")
    
    # Trouver le meilleur threshold basé sur F1-score
    best_result = max(results, key=lambda x: x['f1_score'])
    best_threshold = best_result['threshold']
    
    print("=" * 80)
    print(f"\n✓ MEILLEUR THRESHOLD: {best_threshold:.2f}")
    print(f"  Accuracy:  {best_result['accuracy']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall:    {best_result['recall']:.4f}")
    print(f"  F1-Score:  {best_result['f1_score']:.4f}")
    print(f"  TP={best_result['tp']}, TN={best_result['tn']}, "
          f"FP={best_result['fp']}, FN={best_result['fn']}")
    
    # Sauvegarder les résultats
    output_dir = Path(config.PERFORMANCE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{args.model.lower()}_threshold_optimization.json"
    
    optimization_data = {
        'model': args.model,
        'validation_samples': int(len(mels)),
        'threshold_range': {
            'min': float(args.min),
            'max': float(args.max),
            'step': float(args.step)
        },
        'all_results': results,
        'best_threshold': float(best_threshold),
        'best_metrics': best_result
    }
    
    with open(output_file, 'w') as f:
        json.dump(optimization_data, f, indent=2)
    
    print(f"\n✓ Résultats sauvegardés: {output_file}")
    
    # Retourner le meilleur threshold (pour usage dans scripts)
    return best_threshold


if __name__ == '__main__':
    best_thresh = main()
    # Écrire dans un fichier temporaire pour que le shell puisse le lire
    with open('/tmp/optimized_threshold.txt', 'w') as f:
        f.write(str(best_thresh))
