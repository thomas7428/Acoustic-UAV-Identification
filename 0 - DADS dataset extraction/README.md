# DADS Dataset Extraction

Ce dossier contient les outils pour télécharger et préparer le dataset **DADS (Drone Audio Detection Samples)** depuis Hugging Face.

## À propos de DADS

**DADS** est actuellement la plus grande base de données audio de drones publiquement disponible, spécialement conçue pour développer des systèmes de détection de drones utilisant des techniques de deep learning.

- **Source:** [geronimobasso/drone-audio-detection-samples](https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples)
- **Taille:** 180 320 fichiers audio (~6.81 GB)
- **Classes:**
  - `0` : No drone (16 729 samples)
  - `1` : Drone (163 591 samples)
- **Format original:**
  - Sample rate: 16 kHz
  - Bit depth: 16 bits
  - Canaux: Mono
  - Durée: Variable (0.5s à plusieurs minutes)

## Script d'extraction

Le script `download_and_prepare_dads.py` :
- ✅ Télécharge le dataset depuis Hugging Face
- ✅ Resample à 22 050 Hz (standard du projet)
- ✅ Organise les fichiers par label (`0/` et `1/`)
- ✅ Permet de limiter le nombre de fichiers extraits
- ✅ Affiche la progression et les statistiques

## Installation des dépendances

Avant d'utiliser le script, installez les packages nécessaires :

```powershell
pip install datasets librosa soundfile tqdm
```

Ou si vous utilisez le fichier requirements du projet :
```powershell
pip install -r requirements.txt
```

## Utilisation

### Exemples d'utilisation

**1. Extraction complète (attention : ~180k fichiers, peut prendre des heures)**
```powershell
python "0 - DADS dataset extraction\download_and_prepare_dads.py" --output "0 - DADS dataset extraction\dataset_full"
```

**2. Extraction limitée pour tests rapides (1000 fichiers au total)**
```powershell
python "0 - DADS dataset extraction\download_and_prepare_dads.py" --output "0 - DADS dataset extraction\dataset_test" --max-samples 1000
```

**3. Extraction équilibrée (1000 par classe = 2000 total)**
```powershell
python "0 - DADS dataset extraction\download_and_prepare_dads.py" --output "0 - DADS dataset extraction\dataset_balanced" --max-per-class 1000
```

**4. Petit jeu de données pour développement (100 par classe)**
```powershell
python "0 - DADS dataset extraction\download_and_prepare_dads.py" --output "0 - DADS dataset extraction\dataset_dev" --max-per-class 100
```

### Options disponibles

```
--output, -o          Dossier de sortie pour le dataset (défaut: 'dataset')
--max-samples         Nombre maximum total de fichiers à extraire
--max-per-class       Nombre maximum de fichiers par classe (0 et 1)
--quiet, -q           Supprime l'affichage de progression
```

## Structure de sortie

Après extraction, le dossier aura cette structure :

```
dataset/
├── 0/                          # Classe "No drone"
│   ├── dads_0_000000.wav
│   ├── dads_0_000001.wav
│   └── ...
└── 1/                          # Classe "Drone"
    ├── dads_1_000000.wav
    ├── dads_1_000001.wav
    └── ...
```

Cette structure est compatible avec les scripts de prétraitement existants (`Mel_Preprocess_and_Feature_Extract.py` et `MFCC_Preprocess_and_Feature_Extract.py`).

## Prochaines étapes

Une fois le dataset extrait :

1. **Mettre à jour le chemin dans les scripts de prétraitement**
   
   Éditez `1 - Preprocessing and Features Extraction/Mel_Preprocess_and_Feature_Extract.py` :
   ```python
   DATASET_PATH = "0 - DADS dataset extraction/dataset_balanced"  # ou votre chemin
   ```

2. **Extraire les features**
   ```powershell
   python "1 - Preprocessing and Features Extraction\Mel_Preprocess_and_Feature_Extract.py"
   ```

3. **Entraîner les modèles**
   ```powershell
   python "2 - Model Training\CRNN_Trainer.py"
   ```

## Remarques importantes

- **Temps de téléchargement :** La première exécution télécharge le dataset depuis Hugging Face (~6.81 GB). Les exécutions suivantes utilisent le cache local.
- **Espace disque :** Prévoir suffisamment d'espace (dataset original + versions resamplées).
- **Resampling :** Le script resample automatiquement de 16 kHz → 22.05 kHz pour correspondre au standard du projet.
- **Déséquilibre des classes :** DADS est fortement déséquilibré (10:1 ratio drone:no-drone). Utilisez `--max-per-class` pour créer un jeu équilibré si nécessaire.

## Troubleshooting

**Erreur "datasets module not found"**
```powershell
pip install datasets
```

**Erreur "librosa module not found"**
```powershell
pip install librosa soundfile
```

**Timeout ou erreur réseau**
- Vérifiez votre connexion internet
- Hugging Face peut nécessiter une authentification pour certains datasets (normalement pas pour DADS)

**Manque d'espace disque**
- Utilisez `--max-samples` ou `--max-per-class` pour limiter la taille
- Nettoyez le cache Hugging Face : `~/.cache/huggingface/datasets/`

## Licence

Les données proviennent de sources diverses sous licences Creative Commons et similaires. Consultez la [page du dataset](https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples) pour les détails de licence.
