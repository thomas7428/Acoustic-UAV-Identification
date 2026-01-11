# Threshold Calibration System

Système de calibration hiérarchique multi-critères des seuils de classification.

## Vue d'ensemble

Le système optimise les seuils de classification en utilisant une approche hiérarchique en 3 niveaux:

### Tier 1: Contraintes DURES (must satisfy)
- **Recall > 0.90** : Maximum 10% de faux négatifs (détection manquée)
- **Precision_drone > 0.70** : Minimum 70% de précision pour classe drone (PPV)
- **Precision_ambient > 0.85** : Minimum 85% de précision pour classe ambient (NPV)

### Tier 2: Cible d'optimisation
- **Maximize balanced_precision = min(PPV, NPV)**
- Force l'équilibre entre les deux classes

### Tier 3: Départage
Si égalité sur balanced_precision:
1. Maximiser F1-score
2. Maximiser recall
3. Préférer threshold plus bas (plus permissif)

## Architecture

### Fichiers principaux

```
3 - Single Model Performance Calculation/
├── calibrate_thresholds.py          # Script de calibration principal
├── Universal_Perf_Tester.py         # Charge les thresholds depuis JSON
└── optimize_threshold.py             # (LEGACY - F1 only)

0 - DADS dataset extraction/results/
└── calibrated_thresholds.json       # Thresholds calibrés (centralisé)

6 - Visualization/
└── threshold_analysis.py            # Visualisations des thresholds
```

### Format JSON

Le fichier `calibrated_thresholds.json` contient:

```json
{
  "version": "1.0",
  "calibration_date": "2026-01-10T22:30:00",
  "calibration_mode": "balanced_precision",
  "constraints": {
    "min_recall": 0.90,
    "min_precision_drone": 0.70,
    "min_precision_ambient": 0.85
  },
  "models": {
    "CNN": {
      "threshold": 0.4523,
      "validation_samples": 1234,
      "metrics_at_threshold": {
        "accuracy": 0.8765,
        "precision_drone": 0.7890,
        "precision_ambient": 0.8901,
        "recall": 0.9123,
        "f1_score": 0.8456,
        "balanced_precision": 0.7890,
        "tp": 456, "tn": 678, "fp": 90, "fn": 10
      },
      "constraints_met": {
        "min_recall_0.90": true,
        "min_precision_drone_0.70": true,
        "min_precision_ambient_0.85": true
      },
      "relaxation_applied": null,
      "all_tested_thresholds": [
        {"threshold": 0.05, "f1_score": 0.65, ...},
        ...
      ]
    },
    "RNN": {...},
    "CRNN": {...},
    "ATTENTION_CRNN": {...}
  }
}
```

## Workflow intégré

### 1. Entraînement du modèle
```bash
./run_full_pipeline.sh --models CNN
```

### 2. Calibration automatique
Le pipeline lance automatiquement `calibrate_thresholds.py` pour chaque modèle entraîné:
- Charge les prédictions sur validation set
- Teste tous les thresholds (0.05 à 0.95, step 0.01)
- Applique les critères hiérarchiques
- Sauvegarde dans `calibrated_thresholds.json`

### 3. Évaluation avec thresholds calibrés
`Universal_Perf_Tester.py` charge automatiquement depuis `calibrated_thresholds.json`:
```python
# Priority: CLI arg > calibrated JSON > config.py
if config.CALIBRATION_FILE_PATH.exists():
    calib_data = json.load(...)
    threshold = calib_data['models'][model_name]['threshold']
```

### 4. Visualisations
```bash
cd "6 - Visualization"
python threshold_analysis.py
```

Génère:
- `threshold_analysis_cnn.png` : Courbes métriques vs threshold
- `threshold_analysis_rnn.png`, etc.
- `threshold_comparison_all_models.png` : Comparaison inter-modèles

## Relaxation progressive

Si aucun threshold ne satisfait les contraintes dures, relaxation automatique:

**Level 1**: Relax ambient precision (-5%)
- `min_precision_ambient`: 0.85 → 0.8075

**Level 2**: Relax drone precision (-10% ambient, -5% drone)
- `min_precision_ambient`: 0.85 → 0.765
- `min_precision_drone`: 0.70 → 0.665

**Level 3**: Relax recall (-15% ambient, -10% drone, -5% recall)
- `min_precision_ambient`: 0.85 → 0.7225
- `min_precision_drone`: 0.70 → 0.63
- `min_recall`: 0.90 → 0.855

**Fallback**: Si échec complet → best F1-score (sans contraintes)

## Usage

### Calibrer un modèle spécifique
```bash
python calibrate_thresholds.py --model CNN
```

### Calibrer tous les modèles
```bash
python calibrate_thresholds.py --all-models
```

### Contraintes personnalisées
```bash
python calibrate_thresholds.py --model RNN \
  --min-recall 0.95 \
  --min-precision-drone 0.80 \
  --min-precision-ambient 0.90
```

### Spécifier output custom
```bash
python calibrate_thresholds.py --all-models \
  --output results/my_custom_thresholds.json
```

## Configuration centralisée

Contraintes par défaut dans `config.py`:

```python
CALIBRATION_CONSTRAINTS = {
    'min_recall': 0.90,
    'min_precision_drone': 0.70,
    'min_precision_ambient': 0.85
}

CALIBRATION_FILE_PATH = RESULTS_DIR / "calibrated_thresholds.json"
```

Helper pour charger:
```python
from config import load_calibrated_thresholds

thresholds = load_calibrated_thresholds()
# {'CNN': 0.4523, 'RNN': 0.0123, ...}
```

## Métriques expliquées

### Confusion Matrix
- **TP (True Positive)**: Drone correctement détecté
- **TN (True Negative)**: Ambient correctement identifié
- **FP (False Positive)**: Ambient classé comme drone
- **FN (False Negative)**: Drone manqué (classé comme ambient)

### Métriques dérivées
- **Precision_drone (PPV)**: TP / (TP + FP) - "Si je prédis drone, quelle probabilité que ce soit vrai?"
- **Precision_ambient (NPV)**: TN / (TN + FN) - "Si je prédis ambient, quelle probabilité que ce soit vrai?"
- **Recall (Sensitivity)**: TP / (TP + FN) - "Combien de drones sont détectés?"
- **Specificity**: TN / (TN + FP) - "Combien d'ambients sont correctement identifiés?"
- **F1-Score**: 2 * PPV * Recall / (PPV + Recall) - Moyenne harmonique
- **Balanced_precision**: min(PPV, NPV) - Force équilibre entre classes

## Avantages vs ancien système

### Avant (optimize_threshold.py)
❌ Optimise F1 uniquement (ignore équilibre classes)
❌ Thresholds jamais chargés automatiquement
❌ Édition manuelle de `config.py` requise
❌ RNN threshold = 0.01 (jamais mis à jour)
❌ Pas de validation sur TEST set

### Maintenant (calibrate_thresholds.py)
✅ Optimisation multi-critères hiérarchique
✅ Chargement automatique dans `Universal_Perf_Tester.py`
✅ JSON centralisé, pas de code modifié
✅ Tous les modèles calibrés automatiquement
✅ Relaxation progressive si contraintes trop strictes
✅ Métadonnées complètes pour visualisation
✅ Intégré dans `run_full_pipeline.sh`

## Troubleshooting

### "Calibration file not found"
```bash
# Lancer la calibration manuellement
python "3 - Single Model Performance Calculation/calibrate_thresholds.py" --all-models
```

### "All relaxation levels failed"
Les contraintes sont trop strictes pour ce modèle. Options:
1. Accepter le fallback (best F1)
2. Assouplir les contraintes:
   ```bash
   python calibrate_thresholds.py --model CNN \
     --min-recall 0.85 \
     --min-precision-drone 0.65
   ```

### Visualisations manquantes
```bash
cd "6 - Visualization"
python threshold_analysis.py
```

### Forcer recalibration
```bash
# Supprimer l'ancien fichier
rm "0 - DADS dataset extraction/results/calibrated_thresholds.json"

# Relancer calibration
python "3 - Single Model Performance Calculation/calibrate_thresholds.py" --all-models
```

## Migration depuis ancien système

### Étapes
1. ✅ `calibrate_thresholds.py` créé (remplace `optimize_threshold.py`)
2. ✅ `Universal_Perf_Tester.py` modifié (charge JSON automatiquement)
3. ✅ `config.py` augmenté (constraints + helper function)
4. ✅ `run_full_pipeline.sh` intégré (calibration après training)
5. ✅ `threshold_analysis.py` créé (visualisations)
6. ✅ `run_visualizations.py` mis à jour (inclut threshold analysis)

### Backward compatibility
Le système garde compatibilité avec ancien workflow:
- Si `calibrated_thresholds.json` absent → fallback vers `config.MODEL_THRESHOLDS`
- `optimize_threshold.py` toujours fonctionnel (legacy)
- CLI `--threshold` override toujours prioritaire

## Exemples de visualisations

### Threshold Analysis (par modèle)
- Courbe F1-score vs threshold
- Courbes PPV/NPV vs threshold
- Courbe Recall/Specificity vs threshold
- Pareto front (Precision vs Recall)
- Marquage du point optimal
- Lignes de contraintes

### Model Comparison
- Bar chart des thresholds calibrés
- Bar chart des F1-scores
- Bar chart des balanced precisions
- Bar chart des recalls avec ligne de contrainte

## Développement futur

### Possibles améliorations
- [ ] Calibration sur TEST set (actuellement VAL only)
- [ ] Hot-reload des thresholds sans redémarrage
- [ ] Précalcul des prédictions en .npz (100x plus rapide)
- [ ] Multi-objectif Pareto optimization (scikit-optimize)
- [ ] Coûts FN/FP personnalisés par classe
- [ ] Calibration distance-aware (threshold fonction de la distance)
- [ ] Monitoring drift temporal des thresholds
