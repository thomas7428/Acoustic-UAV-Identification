#!/usr/bin/env python3
"""
Advanced Threshold Optimizer with Interpolation
Trouve le threshold optimal en testant puis en interpolant pour trouver le vrai optimal.
Utilise scipy.optimize pour affiner le threshold entre les points testés.
"""
import json
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from scipy import interpolate
from scipy.optimize import minimize_scalar
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
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def interpolate_f1_curve(thresholds, f1_scores):
    """Crée une fonction d'interpolation cubique pour F1-score."""
    # Interpolation cubique pour lisser la courbe
    f = interpolate.interp1d(thresholds, f1_scores, kind='cubic', 
                             bounds_error=False, fill_value='extrapolate')
    return f


def find_optimal_threshold_refined(model, X, y, test_points):
    """
    Trouve le threshold optimal en 2 étapes:
    1. Teste les points discrets
    2. Interpole et optimise dans la région prometteuse
    """
    # Étape 1: Tester les points discrets
    results = []
    for t in test_points:
        metrics = test_threshold(model, X, y, t)
        results.append(metrics)
    
    thresholds = [r['threshold'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    
    # Trouver le meilleur dans les points testés
    best_discrete = max(results, key=lambda x: x['f1_score'])
    best_idx = f1_scores.index(best_discrete['f1_score'])
    
    # Étape 2: Affiner autour du meilleur point
    # Créer une fonction d'interpolation
    if len(thresholds) >= 4:  # Besoin de 4+ points pour cubic
        f_interp = interpolate_f1_curve(thresholds, f1_scores)
        
        # Définir la plage de recherche autour du meilleur point
        search_min = max(0.0, thresholds[max(0, best_idx-1)])
        search_max = min(1.0, thresholds[min(len(thresholds)-1, best_idx+1)])
        
        # Fonction à minimiser (négatif de F1)
        def neg_f1(t):
            try:
                return -float(f_interp(t))
            except:
                return 0.0
        
        # Optimisation
        result = minimize_scalar(neg_f1, bounds=(search_min, search_max), 
                                method='bounded', options={'xatol': 0.001})
        
        if result.success:
            optimal_t = float(result.x)
            # Tester ce threshold optimal
            optimal_metrics = test_threshold(model, X, y, optimal_t)
            
            print(f"\n[REFINEMENT]")
            print(f"  Best discrete: t={best_discrete['threshold']:.3f}, F1={best_discrete['f1_score']:.4f}")
            print(f"  Optimal refined: t={optimal_t:.3f}, F1={optimal_metrics['f1_score']:.4f}")
            print(f"  Improvement: {(optimal_metrics['f1_score'] - best_discrete['f1_score'])*100:.2f}%")
            
            return optimal_metrics, results
    
    # Si pas assez de points ou échec, retourner le meilleur discret
    return best_discrete, results


def main():
    parser = argparse.ArgumentParser(description='Optimise le threshold avec raffinement')
    parser.add_argument('--model', type=str, required=True,
                        choices=['CNN', 'RNN', 'CRNN', 'ATTENTION_CRNN'],
                        help='Modèle à optimiser')
    parser.add_argument('--min', type=float, default=0.2,
                        help='Seuil minimum à tester (défaut: 0.2)')
    parser.add_argument('--max', type=float, default=0.8,
                        help='Seuil maximum à tester (défaut: 0.8)')
    parser.add_argument('--step', type=float, default=0.05,
                        help='Pas entre chaque seuil (défaut: 0.05)')
    parser.add_argument('--refine', action='store_true',
                        help='Activer le raffinement par interpolation')
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
    print(f"Raffinement: {'Activé' if args.refine else 'Désactivé'}")
    print("=" * 80)
    
    # Créer les points de test
    test_points = np.arange(args.min, args.max + args.step, args.step)
    
    if args.refine:
        # Optimisation avec raffinement
        best_result, all_results = find_optimal_threshold_refined(model, mels, labels, test_points)
    else:
        # Optimisation simple
        all_results = []
        for threshold in test_points:
            metrics = test_threshold(model, mels, labels, threshold)
            all_results.append(metrics)
            print(f"Threshold {threshold:.3f}: Acc={metrics['accuracy']:.4f}, "
                  f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, "
                  f"F1={metrics['f1_score']:.4f}")
        
        best_result = max(all_results, key=lambda x: x['f1_score'])
    
    print("=" * 80)
    print(f"\n✓ MEILLEUR THRESHOLD: {best_result['threshold']:.4f}")
    print(f"  Accuracy:    {best_result['accuracy']:.4f}")
    print(f"  Precision:   {best_result['precision']:.4f}")
    print(f"  Recall:      {best_result['recall']:.4f}")
    print(f"  F1-Score:    {best_result['f1_score']:.4f}")
    print(f"  Specificity: {best_result['specificity']:.4f}")
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
        'refinement_used': args.refine,
        'all_results': all_results,
        'best_threshold': float(best_result['threshold']),
        'best_metrics': best_result
    }
    
    with open(output_file, 'w') as f:
        json.dump(optimization_data, f, indent=2)
    
    print(f"\n✓ Résultats sauvegardés: {output_file}")
    
    # Retourner le meilleur threshold (pour usage dans scripts)
    with open('/tmp/optimized_threshold.txt', 'w') as f:
        f.write(f"{best_result['threshold']:.4f}")
    
    return best_result['threshold']


if __name__ == '__main__':
    best_thresh = main()
