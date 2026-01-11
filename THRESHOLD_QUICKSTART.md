# Quick Start - Threshold Calibration

## Lancer calibration maintenant

```bash
# Calibrer tous les modèles (si déjà entraînés)
python "3 - Single Model Performance Calculation/calibrate_thresholds.py" --all-models

# Ou un seul
python "3 - Single Model Performance Calculation/calibrate_thresholds.py" --model CNN
```

## Voir les résultats

```bash
# Fichier JSON généré
cat "0 - DADS dataset extraction/results/calibrated_thresholds.json"

# Visualiser
cd "6 - Visualization"
python threshold_analysis.py
# → outputs/threshold_analysis_*.png
```

## Utilisation dans le pipeline

```bash
# Le pipeline intègre automatiquement la calibration
./run_full_pipeline.sh --models CNN,RNN

# Ou skip si déjà calibré
./run_full_pipeline.sh --skip-calibration --models CNN
```

## Fichiers importants

- **calibrate_thresholds.py** - Script principal (350 lignes)
- **calibrated_thresholds.json** - Thresholds + métriques (auto-généré)
- **threshold_analysis.py** - Visualisations (280 lignes)
- **README_THRESHOLD_CALIBRATION.md** - Doc complète
- **THRESHOLD_SYSTEM_IMPLEMENTATION.md** - Summary d'implémentation

## Tests

```bash
cd "7 - Tests"
python test_threshold_calibration.py
# → ✓ ALL TESTS PASSED
```

## Optimisation hiérarchique

**Tier 1** (contraintes dures):
- Recall ≥ 0.90
- Precision_drone ≥ 0.70
- Precision_ambient ≥ 0.85

**Tier 2** (optimisation):
- Maximize: balanced_precision = min(PPV, NPV)

**Tier 3** (départage):
- F1 → Recall → Lower threshold

**Relaxation progressive** si contraintes trop strictes (4 niveaux).

## Configuration

Dans `config.py`:
```python
CALIBRATION_CONSTRAINTS = {
    'min_recall': 0.90,
    'min_precision_drone': 0.70,
    'min_precision_ambient': 0.85
}
```

Modifier si besoin ou utiliser CLI:
```bash
python calibrate_thresholds.py --model CNN \
  --min-recall 0.95 \
  --min-precision-drone 0.75
```
