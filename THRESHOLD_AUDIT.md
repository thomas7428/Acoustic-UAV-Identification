# Audit Système de Threshold - Architecture Actuelle et Propositions
**Date**: 2026-01-10

---

## 🔍 ÉTAT DES LIEUX - Architecture Actuelle

### 1. **Sources de Threshold (3 emplacements différents!)**

#### A. `config.py` - Thresholds Hardcodés
```python
MODEL_THRESHOLDS = {
    "CNN": 0.61,
    "RNN": 0.01,  # ⚠️ Valeur suspecte!
    "CRNN": 0.70,
    "Attention_CRNN": 0.77,
}
```

**Problèmes**:
- ❌ Valeurs hardcodées (pas automatiques)
- ❌ RNN threshold = 0.01 semble erroné (probablement jamais mis à jour)
- ❌ Pas de métadonnées (quand/comment calculés?)
- ❌ Pas de versionning

#### B. Fichiers `*_threshold_optimization.json` (par modèle)
**Localisation**: `results/performance/cnn_threshold_optimization.json`

**Contenu**:
```json
{
  "model": "CNN",
  "validation_samples": 2000,
  "threshold_range": {"min": 0.3, "max": 0.7, "step": 0.05},
  "all_results": [...],  // Tous les points testés
  "best_threshold": 0.45,
  "best_metrics": {
    "threshold": 0.45,
    "accuracy": 0.6475,
    "precision": 0.597,
    "recall": 0.911,
    "f1_score": 0.721,
    "tp": 911, "tn": 384, "fp": 616, "fn": 89
  }
}
```

**Avantages**: ✅ Métadonnées complètes, historique des tests
**Problèmes**: 
- ❌ Pas chargé automatiquement par les scripts
- ❌ Un fichier par modèle (pas centralisé)
- ❌ Optimise SEULEMENT F1-score (critère unique)

#### C. `threshold_recommendations.json` (visualisation)
**Localisation**: `6 - Visualization/outputs/threshold_recommendations.json`

**Contenu**:
```json
{
  "UNKNOWN_UNKNOWN": {  // ⚠️ Données invalides!
    "best_f1_threshold": 0.5,
    "best_f1_value": 0,
    "best_acc_threshold": 0.5,
    "balanced_threshold": 0.5
  }
}
```

**Problèmes**:
- ❌ Fichier corrompu (UNKNOWN_UNKNOWN au lieu des modèles)
- ❌ Pas utilisé par les calculateurs de performance
- ❌ Généré par visualisation mais pas par le training

---

### 2. **Scripts d'Optimisation (2 versions redondantes)**

#### A. `optimize_threshold.py` - Version Simple
**Approche**: Grid search avec pas fixe (0.05)

**Workflow**:
```
1. Charger modèle
2. Charger MEL_VAL_DATA (validation set)
3. Tester thresholds de 0.3 à 0.7 par pas de 0.05
4. Calculer métriques pour chaque threshold
5. Sélectionner best_threshold par F1-score max
6. Sauvegarder dans {model}_threshold_optimization.json
```

**Critère d'optimisation**: **F1-score UNIQUEMENT**

**Problèmes**:
- ❌ Un seul critère (F1)
- ❌ Pas de considération des faux négatifs vs faux positifs
- ❌ Pas d'équilibrage drone/ambient
- ❌ Grid search rigide (peut manquer l'optimal entre deux points)

#### B. `optimize_threshold_advanced.py` - Version Avancée
**Approche**: Grid search + interpolation cubique + optimisation

**Workflow**:
```
1. Grid search initial (comme version simple)
2. Identifier région prometteuse (meilleur F1 ± 1 step)
3. Interpolation cubique de la courbe F1(threshold)
4. Optimisation par scipy.minimize_scalar dans la région
5. Affiner le threshold optimal (précision ~0.001)
```

**Critère d'optimisation**: **F1-score UNIQUEMENT** (même problème!)

**Avantages**:
- ✅ Plus précis (interpolation)
- ✅ Trouve l'optimal réel entre les points du grid

**Problèmes**:
- ❌ TOUJOURS un seul critère (F1)
- ❌ Complexité ajoutée pour gain marginal
- ❌ Pas utilisé dans le pipeline automatique

---

### 3. **Utilisation des Thresholds**

#### Dans `Universal_Perf_Tester.py`:
```python
# Ligne 173-178: Résolution du threshold
default_threshold = 0.5
try:
    default_threshold = config.MODEL_THRESHOLDS_NORMALIZED.get(args.model.upper(), 
                                                                config.MODEL_THRESHOLDS.get(args.model, 0.5))
except Exception:
    default_threshold = config.MODEL_THRESHOLDS.get(args.model, 0.5)

resolved_threshold = args.threshold if args.threshold is not None else default_threshold
```

**Comportement actuel**:
1. Si `--threshold` fourni en argument → utilise cette valeur
2. Sinon → utilise `config.MODEL_THRESHOLDS[model]`
3. Fallback → 0.5

**Problèmes**:
- ❌ Ne charge PAS les fichiers `*_threshold_optimization.json`
- ❌ Nécessite mise à jour manuelle de `config.py`
- ❌ Pas de workflow automatique après training

---

### 4. **Workflow Actuel (Cassé)**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. TRAINING                                                 │
│    python CNN_Trainer.py → sauvegarde modèle                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. OPTIMIZATION (MANUEL - rarement fait!)                   │
│    python optimize_threshold.py --model CNN                 │
│    → génère cnn_threshold_optimization.json                 │
│    → best_threshold = 0.45                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. MISE À JOUR CONFIG (MANUEL!)                             │
│    Éditer config.py à la main:                              │
│    MODEL_THRESHOLDS["CNN"] = 0.45  # ← Jamais fait!         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. PERFORMANCE CALCULATION                                  │
│    python Universal_Perf_Tester.py --model CNN              │
│    → utilise config.MODEL_THRESHOLDS["CNN"] = 0.61 (ancien!)│
└─────────────────────────────────────────────────────────────┘
```

**RUPTURE**: Les thresholds optimisés ne sont jamais utilisés!

---

## 🔴 PROBLÈMES CRITIQUES IDENTIFIÉS

### Priority 1: **Critère d'Optimisation Trop Simple**

**Actuel**: Maximise **F1-score uniquement**

**Problème**: F1 = moyenne harmonique de Precision et Recall
- Donne même poids aux FP (faux positifs) et FN (faux négatifs)
- **Pas adapté** si coûts FN >> FP ou vice versa

**Besoin utilisateur** (d'après votre description):
> "maximiser l'efficacité avec des critères d'importances différentes (du plus nécessaire au moins nécessaire) allant de l'écart faible entre la détection de drone et de non drones, le taux de faux négatif, la précision de détection de drones, F1, etc..."

**Traduction**:
1. **Écart faible entre détection drone/non-drone** → Balance precision_drone ≈ precision_ambient (NPV)
2. **Taux de faux négatifs bas** → Recall élevé (minimize FN)
3. **Précision de détection de drones** → Precision élevée (minimize FP)
4. **F1** → Critère composite (dernier recours)

**Ordre de priorité suggéré**:
```
1. HARD CONSTRAINTS (must satisfy):
   - Recall > 0.90 (max 10% FN - drones non détectés)
   - Precision_drone > 0.70 (max 30% FP - fausses alarmes acceptables)

2. OPTIMIZATION TARGET (maximize):
   - Balanced score = min(Precision_drone, NPV_ambient)
   - → Force équilibre entre classes
   
3. TIE-BREAKER (si égalité):
   - F1-score max
```

---

### Priority 2: **Pas de Workflow Automatique**

**Problèmes**:
1. ❌ Optimisation threshold **manuelle** (jamais faite après training)
2. ❌ Mise à jour config.py **manuelle**
3. ❌ Fichiers optimization JSONs **ignorés** par calculateurs
4. ❌ `config.MODEL_THRESHOLDS` **obsolètes**

**Exemple concret**:
```python
# config.py (ligne 195)
"RNN": 0.01  # ← WTF? Threshold à 1%?
             # Probablement jamais optimisé après le premier test
```

**Conséquence**: RNN va classifier TOUT comme drone (prob > 0.01)
→ Précision catastrophique, FP énormes

---

### Priority 3: **Fichier de Stockage Incohérent**

**Actuel**: 3 fichiers différents avec formats différents
1. `config.py` → Python dict hardcodé
2. `results/performance/{model}_threshold_optimization.json` → Détails complets
3. `6 - Visualization/outputs/threshold_recommendations.json` → Corrompu

**Besoin**: **UN SEUL fichier centralisé** chargeable en hot-plug

**Format suggéré**: `results/calibrated_thresholds.json`
```json
{
  "version": "1.0",
  "calibration_date": "2026-01-10T22:00:00",
  "calibration_mode": "balanced_precision",
  "models": {
    "CNN": {
      "threshold": 0.61,
      "validation_set": "val_2000_samples",
      "metrics_at_threshold": {
        "accuracy": 0.6475,
        "precision_drone": 0.597,
        "precision_ambient": 0.850,  // NPV
        "recall": 0.911,
        "specificity": 0.384,
        "f1_score": 0.721,
        "balanced_precision": 0.597  // min(prec_drone, NPV)
      },
      "constraints_met": {
        "min_recall_0.90": true,
        "min_precision_drone_0.70": false  // ⚠️
      },
      "optimization_history": {
        "tested_range": [0.3, 0.7],
        "best_f1_threshold": 0.45,
        "best_balanced_threshold": 0.61,
        "recommended": 0.61
      }
    },
    "RNN": { ... },
    "CRNN": { ... },
    "Attention_CRNN": { ... }
  }
}
```

---

### Priority 4: **Calcul Basé sur VAL Uniquement**

**Actuel**: Optimisation sur **validation set** uniquement

**Problème**: Risque d'overfitting au validation set

**Workflow recommandé**:
```
1. Générer prédictions sur VAL (sans labels révélés au modèle)
2. Optimiser threshold sur VAL
3. Tester threshold optimal sur TEST (indépendant)
4. Si dégradation > 5% → threshold trop overfitté
5. Calculer métriques finales sur TRAIN, VAL, TEST avec threshold optimal
```

**Note**: Vous mentionnez:
> "Le calcul de threshold doit se baser sur des prédictions précalculées sur test et val"

→ **Attention**: Optimiser sur TEST est du **data leakage**!
→ **Solution**: Utiliser TEST seulement pour validation finale

---

## ✅ PROPOSITIONS D'AMÉLIORATION

### Proposition 1: **Optimisation Multi-Critères Hiérarchique**

**Nouveau script**: `calibrate_thresholds_v2.py`

**Critères par ordre de priorité**:

#### **Tier 1: HARD CONSTRAINTS (must satisfy)**
```python
CONSTRAINTS = {
    'min_recall': 0.90,              # Max 10% FN (drones manqués)
    'min_precision_drone': 0.70,     # Max 30% FP (fausses alarmes)
    'min_precision_ambient': 0.85,   # Max 15% erreurs ambient (NPV)
}
```

#### **Tier 2: OPTIMIZATION TARGET**
```python
def balanced_precision_score(precision_drone, precision_ambient):
    """
    Score équilibré: favorise équilibre entre classes.
    Pire classe domine (comme F-beta mais pour les deux classes).
    """
    return min(precision_drone, precision_ambient)
    # Alternative: harmonic mean = 2 / (1/p_drone + 1/p_ambient)
```

#### **Tier 3: TIE-BREAKERS**
```python
if balanced_precision_equal:
    # 1. Favoriser F1 plus élevé
    # 2. Si égalité F1, favoriser recall plus élevé (moins de FN)
    # 3. Si égalité recall, favoriser threshold plus bas (plus permissif)
```

**Pseudo-code**:
```python
def optimize_threshold_hierarchical(predictions_val, labels_val):
    """
    Optimise threshold avec critères hiérarchiques.
    """
    candidates = []
    
    # Tester range de thresholds
    for threshold in np.arange(0.05, 0.95, 0.01):
        metrics = calculate_metrics(predictions_val, labels_val, threshold)
        
        # Tier 1: Vérifier contraintes DURES
        if (metrics['recall'] >= 0.90 and
            metrics['precision_drone'] >= 0.70 and
            metrics['precision_ambient'] >= 0.85):
            
            # Tier 2: Calculer score d'optimisation
            metrics['balanced_precision'] = min(metrics['precision_drone'], 
                                                 metrics['precision_ambient'])
            candidates.append(metrics)
    
    if not candidates:
        # Aucun threshold ne satisfait les contraintes
        # → Relax constraints progressivement
        print("⚠️ Aucun threshold satisfait les contraintes dures!")
        print("   Relaxation des contraintes...")
        return optimize_with_relaxed_constraints(...)
    
    # Tier 2: Trier par balanced_precision (desc)
    candidates.sort(key=lambda x: x['balanced_precision'], reverse=True)
    best = candidates[0]
    
    # Tier 3: Si plusieurs égaux, départager par F1
    ties = [c for c in candidates if c['balanced_precision'] == best['balanced_precision']]
    if len(ties) > 1:
        best = max(ties, key=lambda x: (x['f1_score'], x['recall'], -x['threshold']))
    
    return best['threshold'], best
```

---

### Proposition 2: **Workflow Automatique**

**Intégration dans le pipeline**:

```bash
# 1. TRAINING (inchangé)
python CNN_Trainer.py
# → sauvegarde saved_models/CNN_model.h5

# 2. AUTO-CALIBRATION (nouveau, automatique!)
python calibrate_thresholds_v2.py --model CNN --auto-save
# → génère results/calibrated_thresholds.json
# → met à jour avec threshold optimal

# 3. PERFORMANCE CALCULATION (modifié)
python Universal_Perf_Tester.py --model CNN --use-calibrated
# → charge threshold depuis calibrated_thresholds.json
# → calcule métriques sur train/val/test avec threshold optimal
```

**Modifications requises**:

#### A. `Universal_Perf_Tester.py` - Charger thresholds calibrés
```python
def load_calibrated_threshold(model_name):
    """Charge threshold depuis calibrated_thresholds.json"""
    calib_file = config.CALIBRATION_FILE_PATH  # results/calibrated_thresholds.json
    
    if not calib_file.exists():
        print(f"⚠️ Pas de calibration trouvée: {calib_file}")
        print(f"   Utilisation threshold par défaut depuis config.py")
        return config.MODEL_THRESHOLDS.get(model_name, 0.5)
    
    with open(calib_file, 'r') as f:
        calib_data = json.load(f)
    
    model_data = calib_data.get('models', {}).get(model_name.upper())
    if not model_data:
        print(f"⚠️ Modèle {model_name} non trouvé dans calibration")
        return config.MODEL_THRESHOLDS.get(model_name, 0.5)
    
    threshold = model_data['threshold']
    print(f"✓ Threshold calibré chargé: {threshold:.4f}")
    print(f"  Calibration date: {calib_data['calibration_date']}")
    print(f"  Mode: {calib_data['calibration_mode']}")
    return threshold

# Dans main():
if args.use_calibrated:
    threshold = load_calibrated_threshold(args.model)
else:
    threshold = args.threshold or config.MODEL_THRESHOLDS.get(args.model, 0.5)
```

#### B. `calibrate_thresholds_v2.py` - Script principal
```python
#!/usr/bin/env python3
"""
Threshold Calibration v2 - Multi-Criteria Hierarchical Optimization

Optimise les thresholds en tenant compte de:
1. Contraintes dures (min recall, min precision)
2. Score d'optimisation (balanced precision)
3. Tie-breakers (F1, recall, threshold)

Usage:
    # Calibrer un modèle
    python calibrate_thresholds_v2.py --model CNN
    
    # Calibrer tous les modèles
    python calibrate_thresholds_v2.py --all-models
    
    # Avec sauvegarde automatique
    python calibrate_thresholds_v2.py --all-models --auto-save
"""
```

---

### Proposition 3: **Hot-Plug Capabilities**

**Besoin**: Modifier thresholds sans re-run du code

**Solution**: Watcher de fichier avec reloading automatique

```python
# Dans config.py
import json
from pathlib import Path

class ThresholdManager:
    """Gestionnaire de thresholds avec hot-reload."""
    
    def __init__(self, calib_file_path):
        self.calib_file = Path(calib_file_path)
        self._cache = {}
        self._last_mtime = None
        self._load_if_changed()
    
    def _load_if_changed(self):
        """Recharge si fichier modifié."""
        if not self.calib_file.exists():
            return
        
        mtime = self.calib_file.stat().st_mtime
        if mtime != self._last_mtime:
            with open(self.calib_file, 'r') as f:
                data = json.load(f)
            self._cache = {m: d['threshold'] for m, d in data.get('models', {}).items()}
            self._last_mtime = mtime
            print(f"🔄 Thresholds reloaded from {self.calib_file}")
    
    def get_threshold(self, model_name, default=0.5):
        """Récupère threshold avec auto-reload."""
        self._load_if_changed()  # Check for updates
        return self._cache.get(model_name.upper(), default)

# Instance globale
_threshold_manager = ThresholdManager(CALIBRATION_FILE_PATH)

def get_model_threshold(model_name):
    """API publique pour récupérer threshold."""
    return _threshold_manager.get_threshold(model_name)
```

**Usage**:
```python
# Au lieu de:
threshold = config.MODEL_THRESHOLDS['CNN']

# Utiliser:
threshold = config.get_model_threshold('CNN')
# → Recharge auto si calibrated_thresholds.json modifié
```

---

### Proposition 4: **Validation sur TEST**

**Workflow 2-phase**:

#### Phase 1: Calibration sur VAL
```python
# 1. Charger prédictions VAL (précalculées)
predictions_val = load_predictions(model, 'val')

# 2. Optimiser threshold sur VAL
optimal_threshold = optimize_threshold_hierarchical(predictions_val, labels_val)

print(f"Optimal threshold (VAL): {optimal_threshold}")
```

#### Phase 2: Validation sur TEST
```python
# 3. Appliquer threshold sur TEST (indépendant)
predictions_test = load_predictions(model, 'test')
metrics_test = calculate_metrics(predictions_test, labels_test, optimal_threshold)

# 4. Vérifier dégradation
degradation_f1 = metrics_val['f1'] - metrics_test['f1']
if degradation_f1 > 0.05:
    print("⚠️ Dégradation significative sur TEST!")
    print(f"   VAL F1: {metrics_val['f1']:.4f}")
    print(f"   TEST F1: {metrics_test['f1']:.4f}")
    print(f"   Δ: {degradation_f1:.4f}")
    print("   → Threshold peut être overfitté au VAL set")

# 5. Calculer métriques finales sur TRAIN, VAL, TEST
for split in ['train', 'val', 'test']:
    predictions = load_predictions(model, split)
    metrics = calculate_metrics(predictions, labels, optimal_threshold)
    save_metrics(model, split, threshold, metrics)
```

---

### Proposition 5: **Prédictions Précalculées**

**Motivation**: Éviter de recharger le modèle à chaque optimisation

**Workflow**:

#### Étape 1: Générer prédictions (une fois après training)
```bash
python generate_predictions.py --model CNN --splits train val test
# → sauvegarde results/predictions/cnn_train_predictions.npz
# → sauvegarde results/predictions/cnn_val_predictions.npz
# → sauvegarde results/predictions/cnn_test_predictions.npz
```

**Format NPZ**:
```python
np.savez_compressed(
    'cnn_val_predictions.npz',
    filenames=filenames,     # List[str]
    labels=labels,           # np.array[int]
    probabilities=probs,     # np.array[float] - P(class=1)
    features_shape=(44, 173),
    model_version='1.0'
)
```

#### Étape 2: Optimiser threshold (rapide, sans modèle)
```bash
python calibrate_thresholds_v2.py --model CNN --use-precalc
# → charge cnn_val_predictions.npz (pas besoin du modèle!)
# → optimise threshold rapidement
# → valide sur cnn_test_predictions.npz
```

**Avantages**:
- ✅ Pas besoin de charger TensorFlow
- ✅ Optimisation 100x plus rapide
- ✅ Peut tester plusieurs modes de calibration sans recharger modèle
- ✅ Facilite expérimentation

---

## 📋 PLAN D'IMPLÉMENTATION RECOMMANDÉ

### Phase 1: Fix Urgent (1-2h)
1. **Corriger RNN threshold dans config.py**
   - Lancer optimize_threshold.py pour RNN
   - Mettre à jour config.py avec valeur correcte
   - Vérifier CNN/CRNN/Attention aussi

2. **Créer calibrated_thresholds.json centralisé**
   - Format JSON structuré (voir Proposition 3)
   - Copier meilleurs thresholds depuis optimization JSONs existants
   - Placer dans `results/calibrated_thresholds.json`

3. **Modifier Universal_Perf_Tester pour charger calibrated_thresholds.json**
   - Ajouter flag `--use-calibrated` (default True)
   - Fonction `load_calibrated_threshold()`
   - Fallback vers config.py si fichier absent

### Phase 2: Calibration Multi-Critères (3-4h)
1. **Créer calibrate_thresholds_v2.py**
   - Implémenter optimisation hiérarchique
   - Contraintes dures configurables
   - Score balanced precision
   - Tie-breakers (F1, recall)

2. **Tester sur tous les modèles**
   - Comparer résultats vs optimize_threshold.py
   - Valider que contraintes sont satisfaites
   - Vérifier métriques sur TEST

3. **Générer calibrated_thresholds.json automatiquement**
   - Format complet avec métadonnées
   - Historique d'optimisation
   - Date de calibration

### Phase 3: Prédictions Précalculées (2-3h)
1. **Créer generate_predictions.py**
   - Génère .npz pour train/val/test
   - Sauvegarde dans results/predictions/

2. **Modifier calibrate_thresholds_v2.py**
   - Option `--use-precalc` pour charger .npz
   - Beaucoup plus rapide

3. **Intégrer dans pipeline**
   - run_full_pipeline.sh génère prédictions après training
   - Puis calibration automatique
   - Puis calcul performance

### Phase 4: Hot-Reload (1-2h)
1. **Créer ThresholdManager dans config.py**
   - Watcher de fichier
   - Auto-reload si modifié

2. **API publique get_model_threshold()**
   - Remplace accès direct à MODEL_THRESHOLDS
   - Check mtime à chaque appel

### Phase 5: Documentation et Tests (1-2h)
1. **Documentation complète**
   - README pour calibration
   - Exemples d'usage
   - Explication critères

2. **Tests de validation**
   - Vérifier contraintes satisfaites
   - Comparer VAL vs TEST
   - Vérifier backward compatibility

---

## ⚠️ POINTS D'ATTENTION

### 1. **Backward Compatibility**
- Garder `config.MODEL_THRESHOLDS` comme fallback
- Si `calibrated_thresholds.json` absent → utiliser config.py
- Scripts anciens continuent de fonctionner

### 2. **Contraintes Trop Strictes**
- Si aucun threshold satisfait les contraintes → relaxation progressive
- Alerter utilisateur
- Suggérer de ré-entraîner le modèle ou ajuster contraintes

### 3. **Overfitting au Validation Set**
- TOUJOURS valider sur TEST
- Alerter si dégradation > 5%
- Considérer cross-validation si dataset petit

### 4. **Critères Application-Specific**
- Contraintes actuelles (recall > 0.90, precision > 0.70) sont des **suggestions**
- À adapter selon contexte d'utilisation:
  - **Surveillance critique**: favoriser recall (moins de FN)
  - **Alerte publique**: favoriser precision (moins de FP)

---

## 🎯 QUESTIONS POUR L'UTILISATEUR

Avant implémentation, clarifier:

1. **Ordre de priorité des critères** - Confirmer:
   - Tier 1: Recall > 0.90, Precision_drone > 0.70, NPV_ambient > 0.85?
   - Tier 2: Balanced precision (min des deux)?
   - Tier 3: F1-score comme tie-breaker?

2. **Coût relatif FN vs FP**:
   - FN (drone non détecté): Quel impact? Critique ou acceptable?
   - FP (fausse alarme): Gênant mais acceptable?
   - Ratio importance: FN = 2×FP? 3×FP?

3. **Usage des prédictions précalculées**:
   - Générer après chaque training?
   - Stocker dans results/predictions/?
   - Format NPZ ou JSON?

4. **Workflow automatique**:
   - Calibration automatique après training?
   - Intégrer dans run_full_pipeline.sh?

5. **Hot-reload thresholds**:
   - Nécessaire ou overkill?
   - Check mtime à chaque appel (overhead)?

---

**Prochaine étape**: Attendre vos directives sur les priorités et critères avant implémentation.
