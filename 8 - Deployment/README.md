# 🚁 Raspberry Pi Deployment - Real-Time Drone Detection

Système de détection de drones en temps réel optimisé pour Raspberry Pi.

## 📋 Vue d'ensemble

Ce système permet la détection automatique de drones via l'analyse audio en temps réel.

**✅ DOSSIER AUTONOME** : Tous les modèles et configurations sont inclus. Il suffit de copier ce dossier sur n'importe quelle machine et de lancer `./start_detection.sh`.

## 🚀 Démarrage Rapide

### Installation des dépendances

**Linux / macOS / Raspberry Pi:**
```bash
pip3 install tensorflow librosa numpy soundfile

# Pour l'enregistrement audio (optionnel)
pip3 install pyaudio
# Sur Raspberry Pi/Debian:
sudo apt-get install python3-pyaudio

# Sur Raspberry Pi, privilégier TensorFlow Lite (plus léger):
pip3 install tensorflow-lite librosa numpy soundfile
```

**Windows:**
```powershell
pip install tensorflow librosa numpy soundfile

# Pour l'enregistrement audio (optionnel)
pip install pyaudio
```

**Note Windows**: Si `pyaudio` échoue à installer, télécharger le fichier wheel pré-compilé depuis [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) (choisir la version correspondant à votre Python), puis :
```powershell
pip install PyAudio‑0.2.11‑cp311‑cp311‑win_amd64.whl
```

### Lancer le système

**Linux / macOS / Raspberry Pi:**
```bash
# Mode 1: Détection seule (fichiers manuels)
./start_detection.sh

# Mode 2: Avec enregistrement automatique depuis le micro
./start_detection.sh --with-recording

# Mode 3: Test avec un fichier
./start_detection.sh --test /path/to/audio.wav

# Aide
./start_detection.sh --help
```

**Windows:**
```batch
REM Mode 1: Détection seule (fichiers manuels)
start_detection.bat

REM Mode 2: Avec enregistrement automatique depuis le micro
start_detection.bat --with-recording

REM Mode 3: Test avec un fichier
start_detection.bat --test C:\path\to\audio.wav

REM Aide
start_detection.bat --help
```

C'est tout ! Le script vérifie automatiquement les dépendances, les modèles, et démarre le système.

## 🏗️ Architecture

```
8 - Deployment/
├── start_detection.sh          # 🚀 SCRIPT LANCEMENT Linux/macOS/Raspberry Pi
├── start_detection.bat         # 🚀 SCRIPT LANCEMENT Windows
├── drone_detector.py           # Détecteur (ne pas lancer directement)
├── audio_recorder.py           # Enregistreur (optionnel)
├── deployment_config.json      # Configuration (seuils calibrés)
├── models/                     # ✅ Modèles pré-entraînés INCLUS
│   ├── cnn_model.keras         #    (4.0 MB)
│   ├── rnn_model.keras         #    (8.4 MB)
│   ├── crnn_model.keras        #    (2.4 MB)
│   └── attention_crnn_model.keras  # (15 MB)
├── audio_input/                # Dossier de surveillance
├── logs/                       # Logs et prédictions
└── README.md                   # Ce fichier
```

**✅ Tout est inclus** : Modèles, configuration, scripts. Copier ce dossier suffit.
**✅ Cross-platform** : Compatible Windows, Linux, macOS, Raspberry Pi.

## ⚙️ Configuration

Le fichier `deployment_config.json` contrôle tous les paramètres.

### ✅ Seuils calibrés (déjà configurés)

Les seuils ont été optimisés par calibration class-aware pour maximiser le F1-score :

```json
{
  "detection": {
    "model_thresholds": {
      "CNN": 0.38,              // ✅ Calibré (94.6% accuracy)
      "RNN": 0.51,              // ✅ Calibré (94.9% accuracy)  
      "CRNN": 0.40,             // ✅ Calibré (95.2% accuracy)
      "Attention-CRNN": 0.42    // ✅ Calibré (95.4% accuracy)
    },
    "enabled_models": ["CNN", "Attention-CRNN"],
    "voting_strategy": "majority"
  }
}
```

**⚠️ Important** : Ces seuils sont optimisés pour les données DADS. Ne pas les modifier sans re-calibration.

### Extraction MEL (critique pour performances)

```json
{
  "feature_extraction": {
    "mel_spectrogram": {
      "n_mels": 44,
      "n_fft": 2048,
      "hop_length": 512
    },
    "normalization": false    // ✅ DÉSACTIVÉE pour préserver SNR
  }
}
```

**🔴 Ne jamais activer** `normalization: true` : cela détruirait les différences SNR entre distances et rendrait la détection inefficace.

## 🎯 Utilisation

### Mode 1 : Détection seule (fichiers manuels)

```bash
./start_detection.sh
```

Le système surveille `audio_input/` et analyse automatiquement les fichiers WAV.

**Workflow** :
1. Copier un fichier WAV (4s, 22050Hz, mono) dans `audio_input/`
2. Analyse automatique toutes les 5 secondes
3. Résultat affiché dans console + logs
4. Fichier supprimé après traitement (configurable)

### Mode 2 : Enregistrement + Détection automatique

```bash
./start_detection.sh --with-recording
```

Le système enregistre depuis le micro et analyse en temps réel.

**Pré-requis** : PyAudio installé
```bash
pip3 install pyaudio
# Sur Raspberry Pi:
sudo apt-get install python3-pyaudio
```

### Mode 3 : Test avec un fichier

```bash
./start_detection.sh --test ../dataset/drone_500m.wav
```

Analyse un seul fichier et affiche le résultat détaillé.
2. Le détecteur l'analyse automatiquement
3. Résultat affiché dans la console et logs
4. Fichier supprimé après traitement (configurable)

### Mode 2 : Enregistrement continu

Enregistre depuis le microphone et analyse en temps réel.

**Terminal 1 - Enregistreur** :
```bash
# Lister les devices audio disponibles
python3 audio_recorder.py --list-devices

# Démarrer l'enregistrement continu
python3 audio_recorder.py --interval 5 --duration 4
```

**Terminal 2 - Détecteur** :
```bash
python3 drone_detector.py --continuous
```

### Mode test : Fichier unique

Tester avec un seul fichier audio :

```bash
python3 drone_detector.py --file /path/to/audio.wav
```

## 📊 Sorties

### Console

```
2026-01-08 21:15:42 | WARNING  | 🚨 ALERT | drone_500m.wav | DRONE | Avg Confidence: 89.5% | Votes: 2/2 | 245ms
2026-01-08 21:15:42 | INFO     |     CNN: 87.3% (threshold: 0.38) → DRONE
2026-01-08 21:15:42 | INFO     |     Attention-CRNN: 91.7% (threshold: 0.42) → DRONE

2026-01-08 21:15:47 | INFO     | ✓ CLEAR | ambient_wind.wav | NO_DRONE | Avg Confidence: 12.4% | Votes: 0/2 | 238ms
```

### Fichier de prédictions

`logs/predictions.json` contient l'historique complet :

```json
{
  "timestamp": "2026-01-08 21:15:42",
  "file": "drone_500m.wav",
  "detection": "DRONE",
  "predictions": {
    "CNN": 0.873,
    "Attention-CRNN": 0.917
  },
  "details": {
    "CNN": {
      "probability": 0.873,
      "threshold": 0.38,
      "vote": "DRONE"
    },
    "Attention-CRNN": {
      "probability": 0.917,
      "threshold": 0.42,
      "vote": "DRONE"
    },
    "final_decision": "DRONE",
    "votes_for_drone": 2,
    "total_votes": 2
  },
  "processing_time_ms": 245
}
```

## 🔧 Optimisation Raspberry Pi

### Performances

- **Temps de traitement** : ~200-500ms par fichier (dépend du modèle)
- **RAM** : ~300-500 MB
- **CPU** : 1 core suffit

### Conseils

1. **Modèles recommandés** : CNN (le plus rapide) + Attention-CRNN (le plus précis)
2. **Éviter RNN** : Plus lent, moins précis
3. **Utiliser TensorFlow Lite** : 2-3x plus rapide sur ARM
4. **Limiter les modèles actifs** : 1-2 modèles suffisent

### Configuration optimale pour Raspberry Pi

```json
{
  "detection": {
    "enabled_models": ["CNN"],           // Ou ["CNN", "Attention-CRNN"]
    "voting_strategy": "any"             // Avec 1 modèle
  },
  "performance": {
    "use_gpu": false,
    "batch_size": 1,
    "num_threads": 2,
    "memory_limit_mb": 512,
    "optimize_for_raspberry_pi": true
  }
}
```

## 📝 Exemples d'usage

### Détection en continu avec logs

```bash
python3 drone_detector.py --continuous 2>&1 | tee logs/detector_$(date +%Y%m%d).log
```

### Enregistrement avec device spécifique

```bash
# Trouver le device du microphone USB
python3 audio_recorder.py --list-devices

# Utiliser le device #2
python3 audio_recorder.py --device 2 --interval 5
```

### Test rapide avec fichier

```bash
# Copier un fichier de test
cp "../0 - DADS dataset extraction/dataset_test/1/aug_drone_500m_00001.wav" audio_input/test.wav

# Lancer le détecteur en mode fichier unique
python3 drone_detector.py --file audio_input/test.wav
```

## 🎛️ Stratégies de vote

- **majority** (défaut) : Si la majorité des modèles vote DRONE
- **unanimous** : Tous les modèles doivent voter DRONE
- **any** : Un seul modèle votant DRONE suffit

Choisir selon le compromis précision/rappel souhaité :
- `unanimous` : Moins de faux positifs, peut manquer certains drones
- `any` : Détecte plus de drones, mais plus de faux positifs
- `majority` : Équilibre

## 🔍 Debugging

### Vérifier que les modèles se chargent

```bash
python3 -c "
import tensorflow as tf
model = tf.keras.models.load_model('models/cnn_model.keras')
print('✓ Model loaded:', model.input_shape)
"
```

### Tester l'extraction de features

```bash
python3 -c "
import librosa
import numpy as np
audio, sr = librosa.load('audio_input/test.wav', sr=22050, duration=4)
mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=44)
print('✓ Features shape:', mel.shape)  # Doit être (44, 90)
"
```

### Vérifier les logs

```bash
tail -f logs/detector_$(date +%Y%m%d).log
```

## 🚨 Intégration avec alertes

Le système peut être étendu pour envoyer des alertes :

```python
# Ajouter dans drone_detector.py, fonction log_detection()
if is_drone:
    # Envoyer notification
    os.system("notify-send 'DRONE DETECTED!'")
    
    # Ou appel HTTP
    import requests
    requests.post('http://server/alert', json=result)
    
    # Ou GPIO (LED, buzzer)
    import RPi.GPIO as GPIO
    GPIO.output(LED_PIN, GPIO.HIGH)
```

## 📦 Déploiement complet

### Avec systemd (démarrage automatique)

Créer `/etc/systemd/system/drone-detector.service` :

```ini
[Unit]
Description=Drone Detection System
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/8 - Deployment
ExecStart=/usr/bin/python3 drone_detector.py --continuous
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Activer :

```bash
sudo systemctl daemon-reload
sudo systemctl enable drone-detector
sudo systemctl start drone-detector
sudo systemctl status drone-detector
```

## 🛠️ Maintenance

### Nettoyer les logs anciens

```bash
# Garder seulement les 7 derniers jours
find logs/ -name "detector_*.log" -mtime +7 -delete
```

### Surveiller l'espace disque

```bash
du -sh audio_input/ logs/
```

### Mise à jour des modèles

1. Entraîner nouveaux modèles
2. Copier dans `models/`
3. Redémarrer le détecteur

## 📈 Performances attendues

Avec les seuils calibrés (class-aware optimization) :

| Modèle          | Accuracy | Recall | Precision | F1-Score |
|-----------------|----------|--------|-----------|----------|
| CNN             | 94.6%    | 93.2%  | 96.1%     | 94.6%    |
| RNN             | 94.9%    | 93.9%  | 95.8%     | 94.8%    |
| CRNN            | 95.2%    | 94.1%  | 96.3%     | 95.2%    |
| Attention-CRNN  | 95.4%    | 94.5%  | 96.4%     | 95.4%    |

**Performance par distance** (test set) :

| Distance | Accuracy | Note |
|----------|----------|------|
| 100m     | 96-99%   | Très facile à détecter |
| 500m     | 93-96%   | Bon taux de détection |
| 1000m    | 85-92%   | Plus difficile (SNR faible) |
| Ambient  | 96-99%   | Très peu de faux positifs |

**Faux positifs** : <2% avec seuils calibrés

## ⚡ Troubleshooting

### Le détecteur ne démarre pas

```bash
# Vérifier les dépendances
pip3 list | grep -E "tensorflow|librosa|numpy"

# Vérifier la config
python3 -m json.tool deployment_config.json
```

### Pas de détection

- Vérifier que les fichiers audio sont au bon format (WAV, 22050 Hz, mono, 4 secondes)
- Vérifier les seuils dans la config (peut-être trop élevés)
- Vérifier les logs pour erreurs

### Performances lentes

- Réduire le nombre de modèles actifs
- Utiliser TensorFlow Lite
- Augmenter `scan_interval_seconds`

## 📞 Support

Pour questions ou problèmes, consulter :
- Logs détaillés dans `logs/`
- Documentation du projet principal
- Configuration de référence : `deployment_config.json`

---

**Version** : 1.0  
**Dernière mise à jour** : 14 décembre 2025
