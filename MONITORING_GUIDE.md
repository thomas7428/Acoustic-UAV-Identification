# 📊 Guide des Scripts de Monitoring

Ce projet contient deux scripts de monitoring complémentaires pour suivre l'avancement des entraînements et pipelines.

---

## 🎯 Scripts Disponibles

### 1. `monitor_training.sh` - **Monitoring Focused sur l'Entraînement**

**Usage recommandé** : Pour surveiller **uniquement les entraînements de modèles** en cours.

```bash
bash monitor_training.sh                    # Auto-détecte le dernier log
bash monitor_training.sh logs/phase2e.log   # Log spécifique
```

**Affiche** :
- ✅ Progression de chaque modèle (CNN, RNN, CRNN, Attention-CRNN)
- 📈 Epoch actuel, validation accuracy, validation loss
- ⏱️ Estimation du temps restant (ETA)
- 🔄 Processus actifs (trainers en cours)
- 📝 Activité récente du log

**Avantages** :
- Focus sur les métriques d'entraînement
- Temps réel avec ETA pour chaque modèle
- Détection automatique des trainers actifs
- Léger et rapide (5s de refresh)

**Quand l'utiliser** :
- Phase 2E, 2F, ou tout entraînement standalone
- Quand seuls les modèles sont en cours d'entraînement
- Pour un monitoring léger et précis des métriques

---

### 2. `monitor_pipeline.sh` - **Monitoring Complet du Pipeline**

**Usage recommandé** : Pour surveiller **l'ensemble du pipeline** (dataset + features + training + évaluation + visualisations).

```bash
bash monitor_pipeline.sh                              # Auto-détecte le dernier log
bash monitor_pipeline.sh logs/phase2f_fullpipeline_*.log  # Pipeline complet
```

**Affiche** :
- 📦 Étapes du pipeline (Dataset → Features → Training → Performance → Visualizations)
- ✅ Statut de chaque étape (✓ Completed / ⏳ In Progress / ○ Not Started)
- 📈 Métriques d'entraînement live (via logs individuels)
- 🚨 Erreurs et warnings détectés
- 🔄 Processus actifs (pipeline + trainers)
- 📝 Dernières lignes d'activité

**Avantages** :
- Vue complète de toutes les étapes
- Détection automatique de l'étape courante
- Intégration des métriques de `monitor_training.sh`
- Alerte sur erreurs/warnings
- Résumé final avec chemins des résultats

**Quand l'utiliser** :
- Pipeline complet Phase 2F (master_setup_v2.py)
- Scripts `run_full_pipeline.sh`
- Quand plusieurs étapes se suivent automatiquement
- Pour une vue d'ensemble du projet

---

## 🔄 Différences Clés

| Critère | monitor_training.sh | monitor_pipeline.sh |
|---------|---------------------|---------------------|
| **Focus** | Entraînements uniquement | Pipeline complet |
| **Détails** | Métriques précises par epoch | Vue d'ensemble des étapes |
| **ETA** | Oui (par modèle) | Non (global) |
| **Étapes** | Non (assume training actif) | Oui (6 étapes trackées) |
| **Logs** | Un seul log principal | Multiple logs (pipeline + trainers) |
| **Détection erreurs** | Non | Oui (compte ERROR/WARN) |
| **Use case** | Entraînement standalone | Automatisation complète |

---

## 📋 Exemples d'Utilisation

### Scénario 1 : Lancer Phase 2F Complete

```bash
# Terminal 1 : Lancer le pipeline
cd "0 - DADS dataset extraction"
bash run_full_pipeline.sh augment_config_v3.json

# Terminal 2 : Monitoring complet
cd ..
bash monitor_pipeline.sh
```

**Sortie attendue** :
```
=================================
 COMPLETE PIPELINE MONITOR
=================================
Log: phase2f_fullpipeline_20251214_183045.log
Time: 2025-12-14 18:35:22
Status: RUNNING (PID: 3826665)
Active Trainers: 2
=================================

[CURRENT STEP]
[INFO] Training CRNN model...
[INFO] Training Attention-CRNN model...

[PROGRESS SUMMARY]
  ✓ Step 1: Dataset Generation
  ✓ Step 2: Feature Extraction
  ✓ Step 3a: CNN Training
  ✓ Step 3b: RNN Training
  ⏳ Step 3c: CRNN Training (in progress)
  ⏳ Step 3d: Attention-CRNN Training (in progress)
  ○ Step 4: Performance Calculations
  ○ Step 5: Threshold Calibration
  ○ Step 6: Visualizations

[TRAINING METRICS - LIVE]
    cnn: Epoch 78 | Val Acc: 92.34% | Val Loss: 0.2145
    rnn: Epoch 65 | Val Acc: 89.12% | Val Loss: 0.3021
    crnn: Epoch 23 (in progress...)
    attention_crnn: Epoch 18 (in progress...)

[RECENT ACTIVITY]
[INFO] CRNN - Epoch 23/1000
[INFO] Attention_CRNN - Epoch 18/1000
```

---

### Scénario 2 : Entraîner Uniquement CNN + RNN

```bash
# Terminal 1 : Lancer les trainers
cd "2 - Model Training"
source ../.venv/bin/activate
python CNN_Trainer.py &
python RNN_Trainer.py &

# Terminal 2 : Monitoring focused
cd ..
bash monitor_training.sh
```

**Sortie attendue** :
```
========================================
  TRAINING PROGRESS MONITOR
========================================
Log: phase2e.log
Time: 18:40:15

Stage: 🔄 Training Models

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MODEL TRAINING PROGRESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CNN: Epoch 45 | Val Acc: 91.23% | Val Loss: 0.2345
  ETA: ~15min

RNN: Epoch 32 | Val Acc: 88.67% | Val Loss: 0.3102
  ETA: ~25min

CRNN: Not started

Attention_CRNN: Not started

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RECENT ACTIVITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[INFO] CNN - Epoch 45/1000 - val_accuracy: 0.9123
[INFO] RNN - Epoch 32/1000 - val_accuracy: 0.8867

Active Trainers:
  • CNN_Trainer.py
  • RNN_Trainer.py
```

---

### Scénario 3 : Pipeline Terminé, Vérifier Résultats

```bash
# Monitoring montrera la complétion
bash monitor_pipeline.sh
```

**Sortie finale** :
```
=================================
 PIPELINE COMPLETED!
=================================

Total Duration: 3h 24min 18s

Check results:
  • Visualizations: 6 - Visualization/outputs/
  • Performance: 0 - DADS dataset extraction/results/
  • Models: 0 - DADS dataset extraction/saved_models/

Monitoring stopped.
Full log: logs/phase2f_fullpipeline_20251214_183045.log
```

---

## 🎨 Codes Couleur

Les scripts utilisent des couleurs pour améliorer la lisibilité :

| Couleur | Signification |
|---------|---------------|
| 🟢 **GREEN** | Complété avec succès |
| 🟡 **YELLOW** | En cours / Warning |
| 🔵 **CYAN** | Titres / Info |
| 🔴 **RED** | Erreur / Non démarré |
| ⚪ **NC** | Neutre |

---

## ⚙️ Configuration

Les scripts sont **auto-configurables** :

1. **Détection automatique des logs** : Trouve le log le plus récent si non spécifié
2. **Support multi-logs** : `monitor_pipeline.sh` lit les logs individuels de training
3. **Refresh interval** : 5 secondes (modifiable dans le code)
4. **Format numérique** : `LC_NUMERIC=C` pour compatibilité internationale

---

## 🔧 Personnalisation

### Changer l'intervalle de refresh

```bash
# Dans le script, ligne "sleep 5" → "sleep 10"
sed -i 's/sleep 5/sleep 10/' monitor_training.sh
```

### Ajouter un nouveau modèle

```bash
# Dans monitor_training.sh, ajouter après attention_crnn :
get_training_progress "nouveau_modele"
echo -e "  ETA: ~$(estimate_time_remaining nouveau_modele)"
echo ""
```

### Filtrer les logs affichés

```bash
# Modifier la ligne tail + grep :
tail -10 "$LOG_FILE" | grep -E "\[INFO\]|\[SUCCESS\]" | tail -3
```

---

## 🐛 Troubleshooting

### Problème : "No training log found"

**Solution** :
```bash
# Vérifier les logs disponibles
ls -lh logs/

# Spécifier manuellement
bash monitor_training.sh logs/phase2f_v3_final_20251214.log
```

---

### Problème : "No pipeline log file found"

**Solution** :
```bash
# monitor_pipeline.sh cherche ces patterns :
# - logs/phase2f_fullpipeline_*.log
# - logs/phase2f_v3_*.log
# - logs/phase*.log

# Créer un lien symbolique si besoin
ln -s logs/mon_log.log logs/phase2f_fullpipeline_current.log
bash monitor_pipeline.sh logs/phase2f_fullpipeline_current.log
```

---

### Problème : Métriques non affichées

**Cause** : Logs de training individuels manquants

**Solution** :
```bash
# Vérifier que les trainers génèrent des logs
ls -lh logs/*_training_*.log

# Si absents, les trainers n'ont pas démarré ou utilisent un autre format
```

---

### Problème : ETA incorrect

**Cause** : Estimation basée sur 50 epochs minimum

**Solution** :
```bash
# Modifier dans estimate_time_remaining() :
local min_epochs=50  # Changer selon votre early stopping
```

---

## 📊 Métriques Trackées

### monitor_training.sh

| Métrique | Source | Format |
|----------|--------|--------|
| Epoch | Logs individuels | `Epoch 45/1000` |
| Val Accuracy | grep "val_accuracy" | `92.34%` |
| Val Loss | grep "val_loss" | `0.2345` |
| ETA | Calcul (epochs restants × temps/epoch) | `~15min` |

### monitor_pipeline.sh

| Étape | Pattern | État |
|-------|---------|------|
| Dataset | "Dataset generation completed" | ✓ / ⏳ / ○ |
| Features | "Feature extraction completed" | ✓ / ⏳ / ○ |
| Training CNN | "CNN training completed" | ✓ / ⏳ / ○ |
| Training RNN | "RNN training completed" | ✓ / ⏳ / ○ |
| Training CRNN | "CRNN training completed" | ✓ / ⏳ / ○ |
| Training Attention-CRNN | "Attention_CRNN training completed" | ✓ / ⏳ / ○ |
| Performance | "performance calculated" | ✓ / ⏳ / ○ |
| Thresholds | "Threshold calibration completed" | ✓ / ⏳ / ○ |
| Visualizations | "visualizations generated" | ✓ / ⏳ / ○ |

---

## 🚀 Best Practices

1. **Toujours utiliser dans un terminal dédié** : Ne pas mélanger avec autres commandes
2. **Garder les logs** : Ne pas supprimer pendant monitoring
3. **Vérifier les chemins** : Exécuter depuis la racine du projet
4. **Ctrl+C pour arrêter** : Le monitoring n'affecte pas le pipeline
5. **Double monitoring possible** : Un par terminal si besoin de 2 vues

---

## 📖 Historique

- **v1.0** : `monitor_training.sh` - Monitoring 3 modèles (CNN, RNN, CRNN)
- **v1.5** : Ajout Attention-CRNN support
- **v2.0** : `monitor_phase2f.sh` créé pour pipeline complet
- **v2.1** : Renommé `monitor_pipeline.sh` + intégration métriques training
- **v2.2** : Support multi-patterns de logs + détection erreurs

---

## 🎯 Recommandations Finales

| Situation | Script Recommandé |
|-----------|-------------------|
| **Entraînement seul** | `monitor_training.sh` |
| **Pipeline automatisé** | `monitor_pipeline.sh` |
| **Phase 2E** | `monitor_training.sh` |
| **Phase 2F complete** | `monitor_pipeline.sh` |
| **Debugging modèle** | `monitor_training.sh` |
| **Production run** | `monitor_pipeline.sh` |
| **Quick check** | `monitor_training.sh` |
| **Full overview** | `monitor_pipeline.sh` |

**Astuce** : Vous pouvez lancer les deux simultanément dans deux terminaux pour avoir le meilleur des deux mondes ! 🎯

---

**Version** : 2.2  
**Dernière mise à jour** : 14 décembre 2025  
**Auteur** : Acoustic UAV Identification Team
