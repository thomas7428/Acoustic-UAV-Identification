# 🚁 Raspberry Pi Deployment - Real-Time Drone Detection

Système de détection de drones en temps réel optimisé pour Raspberry Pi.

## 📋 Vue d'ensemble

Ce système permet la détection automatique de drones via l'analyse audio en temps réel. Il peut fonctionner de deux manières :

1. **Mode surveillance de fichiers** : Analyse les fichiers audio déposés dans un dossier
2. **Mode enregistrement continu** : Enregistre depuis le microphone et analyse automatiquement

## 🏗️ Architecture

```
8 - Deployment/
├── drone_detector.py          # Détecteur principal
├── audio_recorder.py           # Enregistreur audio (optionnel)
├── deployment_config.json      # Configuration
├── setup_deployment.sh         # Script de déploiement
├── models/                     # Modèles pré-entraînés (copier ici)
├── audio_input/                # Dossier de surveillance
├── logs/                       # Logs et prédictions
└── README.md                   # Ce fichier
```

## ⚙️ Configuration

Le fichier `deployment_config.json` contrôle tous les paramètres :

### Paramètres clés

```json
{
  "detection": {
    "scan_interval_seconds": 5,        // Intervalle entre analyses
    "enabled_models": ["CNN", "Attention-CRNN"],  // Modèles actifs
    "model_thresholds": {              // Seuils ajustables en temps réel
      "CNN": 0.85,
      "Attention-CRNN": 0.95
    },
    "voting_strategy": "majority"      // majority, unanimous, any
  }
}
```

### Ajustement des seuils en direct

Modifiez `deployment_config.json` et le détecteur rechargera automatiquement la config.

## 🚀 Installation

### 1. Préparer les modèles

Sur la machine d'entraînement :

```bash
cd "8 - Deployment"
chmod +x setup_deployment.sh
./setup_deployment.sh
```

Cela copie les modèles entraînés dans `models/`.

### 2. Transférer sur Raspberry Pi

```bash
# Depuis la machine d'entraînement
scp -r "8 - Deployment" pi@raspberrypi:/home/pi/

# Ou avec USB / réseau
```

### 3. Installer les dépendances sur Raspberry Pi

```bash
# Dépendances système
sudo apt-get update
sudo apt-get install -y python3-pip portaudio19-dev

# Dépendances Python (version légère pour Raspberry Pi)
pip3 install tensorflow-lite librosa numpy soundfile

# Pour l'enregistrement audio (optionnel)
sudo apt-get install python3-pyaudio
```

**Note** : TensorFlow Lite est recommandé pour Raspberry Pi (plus léger).

## 🎯 Utilisation

### Mode 1 : Surveillance de fichiers

Le détecteur surveille le dossier `audio_input/` et analyse automatiquement les nouveaux fichiers.

```bash
python3 drone_detector.py --continuous
```

**Workflow** :
1. Déposez un fichier WAV (4 secondes, 22050 Hz) dans `audio_input/`
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
2025-12-14 18:00:15 | WARNING  | 🚨 ALERT | recording_20251214_180015.wav | DRONE | Avg Confidence: 89.50% | Votes: 2/2 | 245ms
2025-12-14 18:00:15 | INFO     |     CNN: 87.30% (threshold: 0.85) → DRONE
2025-12-14 18:00:15 | INFO     |     Attention-CRNN: 91.70% (threshold: 0.95) → NO_DRONE
```

### Fichier de prédictions

`logs/predictions.json` contient l'historique complet :

```json
{
  "timestamp": "2025-12-14 18:00:15",
  "file": "recording_20251214_180015.wav",
  "detection": "DRONE",
  "predictions": {
    "CNN": 0.873,
    "Attention-CRNN": 0.917
  },
  "details": {
    "CNN": {
      "probability": 0.873,
      "threshold": 0.85,
      "vote": "DRONE"
    },
    "final_decision": "DRONE",
    "votes_for_drone": 1,
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

Avec les modèles Phase 2F (calibrated thresholds) :

| Distance | Précision | Rappel | F1-Score |
|----------|-----------|--------|----------|
| 500m     | 82-93%    | 85-95% | 84-94%   |
| 350m     | 95-100%   | 95-100%| 97-100%  |
| 200m     | 95-100%   | 95-100%| 97-100%  |
| Ambient  | 98-100%   | 98-100%| 99-100%  |

**Faux positifs** : <2% (avec thresholds calibrés)

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
