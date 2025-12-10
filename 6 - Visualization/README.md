# Visualization Suite

Ce dossier contient les scripts de visualisation et d'analyse pour le projet Acoustic UAV Identification.

## Scripts disponibles

### 1. `dataset_analysis.py`
Analyse la composition et les caractéristiques du dataset.

**Génère :**
- Distribution des classes (original, augmenté, combiné)
- Distribution SNR des échantillons augmentés
- Exemples de formes d'onde et spectrogrammes
- Statistiques récapitulatives (JSON)

**Usage :**
```bash
python dataset_analysis.py
```

### 2. `model_performance.py`
Compare les performances des différents modèles (CNN, RNN, CRNN).

**Génère :**
- Courbes d'apprentissage (accuracy et loss)
- Matrices de confusion
- Comparaison des métriques (accuracy, precision, recall, F1)
- Tableau récapitulatif des performances (CSV)

**Usage :**
```bash
python model_performance.py
```

### 3. `augmentation_impact.py`
Analyse l'impact de l'augmentation des données sur les performances.

**Génère :**
- Performance par catégorie SNR
- Composition du dataset augmenté
- Évolution du dataset (avant/après augmentation)

**Usage :**
```bash
python augmentation_impact.py
```

### 4. `performance_by_distance.py`
Analyse les performances en fonction de la distance simulée du drone (catégories SNR).

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
