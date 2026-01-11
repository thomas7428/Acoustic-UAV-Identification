# Guide de Test Windows - Micro PC Portable

## Installation rapide (Windows)

1. **Ouvrir PowerShell ou CMD**

2. **Installer les dépendances** :
```powershell
pip install tensorflow librosa numpy soundfile pyaudio
```

**Si pyaudio échoue** :
- Télécharger le wheel depuis : https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
- Choisir selon votre version Python (ex: `PyAudio‑0.2.11‑cp311‑cp311‑win_amd64.whl` pour Python 3.11 64-bit)
- Installer : `pip install PyAudio‑0.2.11‑cp311‑cp311‑win_amd64.whl`

## Test rapide avec un fichier

```batch
cd "8 - Deployment"
start_detection.bat --test audio_input\test_drone.wav
```

**Résultat attendu** :
```
🚨 ALERT | test_drone.wav | DRONE | Avg Confidence: 92.22%
```

## Test avec le microphone de votre PC

### Étape 1 : Lister les périphériques audio

```batch
start_detection.bat --list-devices
```

Notez le numéro de votre microphone (ex: `[2] Microphone (Realtek)`)

### Étape 2 : Modifier le fichier de config (optionnel)

Ouvrir `deployment_config.json` et modifier :
```json
{
  "recording": {
    "enabled": true,
    "device_index": 2,   // ← Remplacer par votre numéro
    "sample_rate": 22050,
    "duration_seconds": 4
  }
}
```

### Étape 3 : Lancer avec enregistrement

**Option A - Mode automatique (recommandé)** :
```batch
start_detection.bat --with-recording
```

Le système enregistre depuis votre micro toutes les 5 secondes et analyse automatiquement.

**Option B - Mode manuel (2 fenêtres)** :

Terminal 1 - Enregistreur :
```batch
python audio_recorder.py --interval 5 --duration 4
```

Terminal 2 - Détecteur :
```batch
python drone_detector.py --continuous
```

### Test de détection

1. Lancer le système avec votre micro
2. Faire un bruit de drone :
   - Moteur de drone jouet
   - Vidéo YouTube de drone sur haut-parleur
   - Son de ventilateur/moteur proche
3. Observer la console :
   - `🚨 ALERT ... DRONE` = Détection positive
   - `✓ CLEAR ... NO_DRONE` = Pas de drone

## Arrêter le système

Appuyer sur `Ctrl+C` dans la console

## Vérifier les résultats

Les détections sont sauvegardées dans :
- **Console** : Affichage temps réel
- **logs\detector_YYYYMMDD.log** : Historique complet
- **logs\predictions.json** : Détails JSON (probabilités, votes)

Exemple de log :
```
2026-01-08 21:15:42 | WARNING  | 🚨 ALERT | recording_20260108_211542.wav | DRONE
    CNN: 87.3% (threshold: 0.38) → DRONE
    Attention-CRNN: 91.7% (threshold: 0.42) → DRONE
```

## Dépannage Windows

### Erreur "No module named 'tensorflow'"
```powershell
pip install tensorflow
```

### Erreur pyaudio "Microsoft Visual C++ required"
Télécharger le wheel pré-compilé (voir installation ci-dessus)

### Microphone non détecté
1. Vérifier que le micro fonctionne (Paramètres Windows > Son)
2. Lister les devices : `start_detection.bat --list-devices`
3. Modifier `device_index` dans `deployment_config.json`

### Pas de détection avec bruit ambiant
**Normal** ! Le système détecte des drones, pas n'importe quel bruit :
- Fréquences spécifiques : 1-4 kHz (moteurs de drone)
- Patterns temporels : harmoniques caractéristiques
- SNR : Signal drone vs ambiant

Pour tester correctement :
- Utiliser un vrai drone
- Ou fichiers audio de test fournis (`test_drone.wav`)
- Ou vidéo YouTube de drone avec bon haut-parleur

### Performance lente
1. Utiliser uniquement CNN (plus rapide) :
   - Modifier `deployment_config.json` : `"enabled_models": ["CNN"]`
2. Réduire la fréquence d'analyse :
   - `"scan_interval_seconds": 10` (au lieu de 5)

## Performances attendues sur PC Windows

- **Temps d'analyse** : 200-500ms par fichier (selon CPU)
- **RAM** : ~500 MB
- **CPU** : Intel i5 ou équivalent recommandé
- **Précision** : 94-95% (identique Linux)

## Fichiers générés

```
8 - Deployment/
├── audio_input/
│   └── recording_YYYYMMDD_HHMMSS.wav  (si enregistrement actif)
├── logs/
│   ├── detector_YYYYMMDD.log          (logs du jour)
│   ├── predictions.json               (1000 dernières détections)
│   └── recorder.log                   (logs enregistreur)
```

## Nettoyage

Supprimer les anciens enregistrements :
```batch
del /q audio_input\*.wav
```

Nettoyer les vieux logs :
```batch
forfiles /p logs /s /m *.log /d -7 /c "cmd /c del @path"
```

---

**Tout fonctionne ?** Parfait ! Le système est prêt pour un déploiement réel.

**Des questions ?** Consulter [README.md](README.md) pour la documentation complète.
