# Visualization Suite

Ce dossier contient les scripts de visualisation et d'analyse pour le projet Acoustic UAV Identification.

## 🚀 Quick Start

### Usage Recommandé (Simple et Rapide)

```bash
# Pipeline complet de visualisations
python run_visualizations.py

# Pipeline rapide (sans audio examples ni threshold calibration)
python quick_viz.py fast

# Seulement la comparaison de performances
python quick_viz.py performance

# Pipeline sans audio examples (plus rapide)
python run_visualizations.py --skip-audio
```

---

## 📋 Scripts Principaux

### 🎯 1. `run_visualizations.py` ⭐ **RUNNER UNIFIÉ**
**Point d'entrée principal pour générer toutes les visualisations.**

Pipeline complet en 8 étapes:
1. Select Best Results (génère best_results_summary.json)
2. Performance Comparison (meilleurs thresholds)
3. Threshold Calibration Comparison
4. Model Comparison Plots
5. SNR Distribution Analysis
6. Dataset Composition Analysis
7. Modern Threshold Calibration
8. HTML Report Generation

**Options:**
```bash
python run_visualizations.py                    # Complet
python run_visualizations.py --skip-audio       # Sans audio examples
python run_visualizations.py --skip-threshold   # Sans threshold calibration
```

**Avantages:**
- ✅ Pipeline automatique complet
- ✅ Gestion d'erreurs robuste
- ✅ Résumé clair des résultats
- ✅ Inclut select_best_results automatiquement

---

### ⚡ 2. `quick_viz.py` - Lanceur de Presets
**Raccourcis pour les cas d'usage courants.**

**Presets disponibles:**
- `all` - Pipeline complet
- `fast` - Rapide (sans audio/threshold)
- `performance` - Seulement performances
- `no-audio` - Complet sans audio

**Usage:**
```bash
python quick_viz.py all           # Pipeline complet
python quick_viz.py fast          # Rapide
python quick_viz.py performance   # Juste performances
```

---

## 🎨 Scripts de Visualisation Individuels
**Analyse de la composition et statistiques du dataset.**

Visualise la distribution des classes, sous-catégories et splits.

**Génère :**
- `dataset_split_distribution.png` - Distribution par split
- `dataset_drone_distribution.png` - Drones par distance
- `dataset_ambient_distribution.png` - Ambients par type
- `dataset_summary.txt` - Rapport complet

**Usage :**
```bash
python modern_dataset_analysis.py
```

---

### 3. `modern_dataset_analysis.py`
**Génère des exemples audio représentatifs avec visualisations.**

Crée une page HTML interactive avec lecteurs audio, waveforms et spectrogrammes.

**Génère :**
- Fichiers WAV copiés
- Visualisations (waveform + spectrogramme)
- Page HTML avec lecteurs audio
- Dossier : `outputs/audio_examples/`

**Usage :**
```bash
python modern_audio_examples.py
# Ouvrir: outputs/audio_examples/index.html
```

---

### 4. `modern_audio_examples.py`
**Analyse systématique des thresholds de décision.**

Recommande les thresholds optimaux par modèle pour différents critères (F1, accuracy, équilibrage FP/FN).

**Génère :**
- `threshold_calibration_{model}_{split}.png` - Courbes et recommandations
- `threshold_recommendations.json` - Thresholds optimaux
- `threshold_recommendations.txt` - Rapport lisible

**Usage :**
```bash
python modern_threshold_calibration.py
# Nécessite plusieurs thresholds testés avec Universal_Perf_Tester.py
```

---

### 5. `modern_threshold_calibration.py`
python run_all_visualizations.py --skip-audio   # Sans audio examples
```

---

### 6. `threshold_calibration_comparison.py`
Analyse l'impact des thresholds sur les performances.

**Génère:** `threshold_calibration_comparison.png`

---

### 7. `model_comparison_plots.py`
Comparaisons visuelles entre modèles.

**Génère:** `model_performance_comparison.png`

---

### 8. `snr_distribution.py`
Analyse de la distribution SNR.

**Génère:** `snr_distribution.png`

---

### 9. `generate_html_report.py`
Génère un rapport HTML interactif avec toutes les visualisations.

**Génère:**
- `performance_report.html`
- `images/` (dossier avec toutes les images)

**Note:** Utilise maintenant des chemins relatifs au lieu de base64 (~10x plus léger).

---

## 📁 Organisation

```
6 - Visualization/
├── run_visualizations.py          ⭐ Runner unifié (PRINCIPAL)
├── quick_viz.py                   🚀 Presets rapides
├── select_best_results.py         🎯 Sélection best results
├── performance_comparison_best.py 📊 Comparaison performances
├── modern_dataset_analysis.py     📈 Analyse dataset
├── modern_audio_examples.py       🎵 Exemples audio
├── modern_threshold_calibration.py 🎚️ Calibration thresholds
├── threshold_calibration_comparison.py
├── model_comparison_plots.py
├── snr_distribution.py
├── generate_html_report.py        🌐 Rapport HTML
├── README.md                      📖 Documentation
├── outputs/                       💾 Résultats
│   ├── *.png
│   ├── *.txt
│   ├── *.json
│   ├── performance_report.html
│   ├── images/                    (pour le HTML report)
│   └── audio_examples/
└── _deprecated/                   🗄️ Scripts obsolètes
    ├── _old_run_all_visualizations.py
    ├── _old_run_enhanced_visualizations.py
    └── _deprecated_performance_comparison.py

```

---

## 🔄 Migration depuis l'Ancien Système

### Si vous utilisiez `run_all_visualizations.py`:
```bash
# Ancien:
python run_all_visualizations.py

# Nouveau:
python run_visualizations.py
```

### Si vous utilisiez `run_enhanced_visualizations.py`:
```bash
# Ancien:
python run_enhanced_visualizations.py

# Nouveau:
python run_visualizations.py
```

### Si vous utilisiez `performance_comparison.py --all`:
```bash
# Ancien:
python performance_comparison.py --all

# Nouveau:
python quick_viz.py all
# ou
python run_visualizations.py
```

---

## 🚀 Workflow Rapide

### Générer des résultats de performance

```bash
# 1. Tester un modèle (génère JSON)
python "3 - Single Model Performance Calculation/Universal_Perf_Tester.py" \
    --model CNN --split test --threshold 0.5

# 2. Visualiser immédiatement
cd "6 - Visualization"
python quick_viz.py all
```

### Analyse complète

```bash
# Générer toutes les visualisations
python run_all_visualizations.py

# Résultats dans outputs/
ls outputs/
```

### Comparaisons personnalisées

```bash
# Comparer CNN vs CRNN
python performance_comparison.py --models CNN CRNN --splits test

# Analyser l'impact du threshold
python performance_comparison.py --models CNN --thresholds 0.4 0.5 0.6 0.7
```

---

## 📖 Documentation Complète

- **`README.md`** (ce fichier) - Vue d'ensemble et référence des scripts
- **`WORKFLOW.md`** - Guide détaillé étape par étape avec cas d'usage
- **`archives/README.md`** - Information sur les scripts legacy

---

## 💡 Tips

1. **Utiliser les JSON précalculés** : Tous les scripts modernes lisent depuis `config.PERFORMANCE_DIR`, donc testez vos modèles une fois avec `Universal_Perf_Tester.py`, puis visualisez à volonté sans recalcul.

2. **Presets rapides** : `quick_viz.py` offre des configurations prêtes à l'emploi pour les cas d'usage courants.

3. **Filtrage intelligent** : `performance_comparison.py` peut combiner train/val/test ou comparer différents thresholds automatiquement.

4. **Timestamps automatiques** : Les fichiers JSON incluent un timestamp, donc plusieurs tests du même modèle ne s'écrasent jamais.

5. **run_all_visualizations** : Génère toutes les visualisations essentielles en une commande.

---

## 🔧 Dépannage

**"No JSON files found"**
- Lancer `Universal_Perf_Tester.py` d'abord pour générer les résultats

**"No multi-threshold results"**
- Pour l'analyse de thresholds, tester avec plusieurs valeurs (0.4, 0.5, 0.6, etc.)

**Erreur d'import**
- Vérifier que vous exécutez depuis le virtualenv : `.venv/bin/python`

---

## 📊 Outputs Générés

Tous les résultats sont sauvegardés dans `outputs/` :

**Visualisations PNG:**
- Performance globale et par classe
- Matrices de confusion
- Courbes par distance/type
- Impact des thresholds

**Rapports Texte:**
- `performance_summary.txt` - Métriques détaillées
- `dataset_summary.txt` - Stats du dataset
- `threshold_recommendations.txt` - Thresholds optimaux

**Données JSON:**
- Résultats bruts pour post-traitement

**Pages HTML:**
- `audio_examples/index.html` - Exemples audio interactifs

---

**Génère :**
- Spectre de difficulté de détection (très loin → très proche)
- Courbes théoriques de performance vs distance
- Table d'analyse par catégorie de distance
- Distribution des échantillons par difficulté

**Usage :**
```bash
python performance_by_distance.py
```

**Note :** Ce script génère des courbes théoriques. Pour des performances réelles par catégorie, il faudrait évaluer les modèles sur un test set avec labels de catégorie SNR.

### 5. `run_all_visualizations.py`
Lance tous les scripts de visualisation en une seule commande.

**Usage :**
```bash
python run_all_visualizations.py
```

## Structure des sorties

Toutes les visualisations sont sauvegardées dans le dossier `outputs/` :

```
outputs/
├── dataset_distribution.png
├── snr_distribution.png
├── audio_examples.png
├── dataset_statistics.json
├── training_curves.png
├── confusion_matrices.png
├── metrics_comparison.png
├── performance_table.csv
├── snr_performance.png
├── augmentation_composition.png
├── dataset_evolution.png
├── difficulty_spectrum.png
├── performance_vs_distance.png
└── distance_analysis.csv
```

## Dépendances

Les scripts nécessitent les bibliothèques suivantes :
- matplotlib
- seaborn
- numpy
- pandas
- librosa
- scikit-learn

Ces dépendances sont normalement déjà installées avec le projet principal.

## Ordre d'exécution recommandé

1. **Après génération du dataset :** `dataset_analysis.py`
2. **Après entraînement des modèles :** `model_performance.py`
3. **Pour analyse complète :** `augmentation_impact.py`
4. **Pour tout générer :** `run_all_visualizations.py`

## Notes

- Les scripts utilisent le fichier `config.py` à la racine du projet pour les chemins
- Si un fichier de données est manquant, le script affiche un avertissement mais continue
- Les graphiques sont sauvegardés en haute résolution (300 DPI) pour publication
- Format des sorties : PNG pour les images, JSON/CSV pour les données

## Exemple de workflow complet

```bash
# 1. Générer le dataset et entraîner les modèles
cd "../0 - DADS dataset extraction"
python master_setup.py --complete --max-per-class 500 --augment

cd "../1 - Preprocessing and Features Extraction"
python Mel_Preprocess_and_Feature_Extract.py
python MFCC_Preprocess_and_Feature_Extract.py

cd "../2 - Model Training"
python CNN_Trainer.py
python RNN_Trainer.py
python CRNN_Trainer.py

# 2. Générer toutes les visualisations
cd "../6 - Visualization"
python run_all_visualizations.py
```

## Personnalisation

Pour personnaliser les visualisations, modifiez les paramètres au début de chaque script :
- Style matplotlib/seaborn
- Tailles des figures
- Couleurs
- Polices

## Support

En cas de problème, vérifiez que :
1. Le dataset est bien généré dans `0 - DADS dataset extraction/`
2. Les modèles sont entraînés et les résultats sont dans `results/`
3. Les chemins dans `config.py` sont corrects
4. Toutes les dépendances sont installées
