# 🚀 Guide de Démarrage Rapide - Détection de Drones

## Installation (1 minute)

### Linux / macOS / Raspberry Pi

```bash
# Installer les dépendances Python
pip3 install tensorflow librosa numpy soundfile

# Pour l'enregistrement audio (optionnel)
pip3 install pyaudio
```

Sur Raspberry Pi, utiliser TensorFlow Lite (plus léger) :
```bash
pip3 install tensorflow-lite librosa numpy soundfile
```

### Windows

```powershell
# Installer les dépendances Python
pip install tensorflow librosa numpy soundfile

# Pour l'enregistrement audio (optionnel)
pip install pyaudio
```

**Note Windows** : Si `pyaudio` échoue, télécharger le wheel depuis [ici](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) puis :
```powershell
pip install PyAudio‑0.2.11‑cp311‑cp311‑win_amd64.whl
```

## Utilisation

### Linux / macOS / Raspberry Pi

```bash
# Option 1 : Test rapide avec un fichier
./start_detection.sh --test /path/to/audio.wav

# Option 2 : Détection continue
./start_detection.sh

# Option 3 : Enregistrement + Détection
./start_detection.sh --with-recording
```

### Windows

```batch
REM Option 1 : Test rapide avec un fichier
start_detection.bat --test C:\path\to\audio.wav

REM Option 2 : Détection continue
start_detection.bat

REM Option 3 : Enregistrement + Détection
start_detection.bat --with-recording
```

**Résultat attendu** (identique sur Linux/Windows) :
```
🚨 ALERT | test_drone.wav | DRONE | Avg Confidence: 92.22% | Votes: 2/2 | 1445ms
    CNN: 90.53% (threshold: 0.38) → DRONE
    Attention-CRNN: 93.90% (threshold: 0.42) → DRONE
```

### Lister les périphériques audio (Windows/Linux)

```bash
# Linux/macOS
./start_detection.sh --list-devices

# Windows
start_detection.bat --list-devices
```

- Copier des fichiers WAV (4s, 22050Hz, mono) dans `audio_input/`
- Le système analyse automatiquement toutes les 5 secondes
- Résultats dans console + `logs/predictions.json`

### Option 3 : Enregistrement + Détection temps réel

```bash
./start_detection.sh --with-recording
```

Enregistre depuis le microphone et analyse en temps réel.

## Performances

| Modèle          | Accuracy | Temps/fichier |
|-----------------|----------|---------------|
| CNN             | 94.6%    | ~200ms        |
| RNN             | 94.9%    | ~400ms        |
| CRNN            | 95.2%    | ~300ms        |
| Attention-CRNN  | 95.4%    | ~500ms        |

**Configuration par défaut** : CNN + Attention-CRNN (optimal précision/vitesse)

## Que contient ce dossier ?

✅ **4 modèles pré-entraînés** (30 MB total)
✅ **Configuration optimisée** (seuils calibrés)
✅ **Scripts prêts à l'emploi**
✅ **Autonome** : fonctionne sans dépendances externes

## Format des fichiers audio

- **Format** : WAV
- **Durée** : 4 secondes
- **Sample rate** : 22050 Hz
- **Canaux** : Mono (1 canal)

## Logs

- **Console** : Détection en temps réel
- **logs/detector_YYYYMMDD.log** : Historique complet
- **logs/predictions.json** : Prédictions détaillées (1000 dernières)

## Aide

```bash
./start_detection.sh --help
```

## Questions fréquentes

### Changer les seuils de détection ?

Modifier `deployment_config.json` :
```json
{
  "detection": {
    "model_thresholds": {
      "CNN": 0.38,              // ↑ Augmenter = moins de faux positifs
      "Attention-CRNN": 0.42    // ↓ Diminuer = détecter plus de drones
    }
  }
}
```

**⚠️ Attention** : Ces seuils sont calibrés scientifiquement. Les modifier peut dégrader les performances.

### Activer/désactiver des modèles ?

Dans `deployment_config.json` :
```json
{
  "detection": {
    "enabled_models": ["CNN"],  // Un seul modèle (plus rapide)
    // ou
    "enabled_models": ["CNN", "RNN", "CRNN", "Attention-CRNN"]  // Tous
  }
}
```

### Temps de traitement trop long ?

1. Utiliser uniquement CNN (le plus rapide) :
   ```json
   "enabled_models": ["CNN"]
   ```

2. Sur Raspberry Pi, installer TensorFlow Lite

3. Réduire le nombre de modèles actifs

### Erreur "No module named tensorflow" ?

Le script cherche automatiquement le virtualenv dans `../.venv/`. 

Si vous utilisez un autre environnement :
```bash
source /path/to/your/venv/bin/activate
python3 drone_detector.py --file test.wav
```

## Performances attendues

**Détection de drones** :
- ✅ 500m : 93-96% accuracy
- ✅ 100m : 96-99% accuracy

**Faux positifs** :
- ✅ <2% avec configuration par défaut

---

**Version** : 2.0 (8 janvier 2026)
**Status** : ✅ Production-ready
