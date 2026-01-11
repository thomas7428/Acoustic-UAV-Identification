╔══════════════════════════════════════════════════════════════════════════╗
║                   SYSTÈME DE DÉTECTION DE DRONES                         ║
║                         GUIDE WINDOWS                                     ║
╚══════════════════════════════════════════════════════════════════════════╝

🚀 DÉMARRAGE ULTRA-RAPIDE (3 étapes)

1. Installer les dépendances Python
   ─────────────────────────────────
   Ouvrir PowerShell ou CMD et taper:
   
   pip install tensorflow librosa numpy soundfile
   
   Si pyaudio échoue (pour micro):
   → Télécharger wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
   → Installer: pip install PyAudio-0.2.11-cp311-cp311-win_amd64.whl

2. Tester avec un fichier
   ───────────────────────
   start_detection.bat --test audio_input\test_drone.wav
   
   Résultat attendu:
   🚨 ALERT | test_drone.wav | DRONE | Avg Confidence: 92.22%

3. Tester avec votre micro
   ────────────────────────
   start_detection.bat --with-recording
   
   Parler près du micro ou lancer une vidéo de drone
   Observer la console pour détections

📁 FICHIERS IMPORTANTS

  start_detection.bat      → Script principal Windows
  WINDOWS_GUIDE.md         → Guide détaillé Windows
  QUICKSTART.md            → Guide multi-plateforme
  README.md                → Documentation complète
  test_quick_windows.bat   → Test automatisé rapide

🎯 MODES D'UTILISATION

  Test fichier unique:
    start_detection.bat --test C:\path\to\audio.wav
  
  Surveillance dossier (fichiers manuels):
    start_detection.bat
    → Déposer WAV dans audio_input\
  
  Enregistrement micro + détection:
    start_detection.bat --with-recording
    → Enregistre toutes les 5s et analyse

📊 RÉSULTATS

  Console:     Affichage temps réel
  logs\:       Historique complet (detector_YYYYMMDD.log)
  logs\:       JSON détaillé (predictions.json)

⚠️ IMPORTANT

  Format audio requis:
    • WAV mono (1 canal)
    • 22050 Hz
    • 4 secondes

  Performances attendues:
    • Accuracy: 94-95%
    • Temps: 200-500ms par fichier
    • Faux positifs: <2%

🔧 DÉPANNAGE RAPIDE

  Erreur "No module named..."
    → pip install tensorflow librosa numpy soundfile
  
  Pas de détection avec bruit ambiant
    → Normal! Détecte UNIQUEMENT les drones
    → Tester avec fichiers fournis ou vrai drone
  
  Micro non détecté
    → start_detection.bat --list-devices
    → Modifier device_index dans deployment_config.json

📖 DOCUMENTATION COMPLÈTE

  WINDOWS_GUIDE.md    → Guide étape par étape Windows
  QUICKSTART.md       → Démarrage rapide multi-OS
  README.md           → Documentation technique complète

✅ COMPATIBILITÉ

  ✓ Windows 10/11
  ✓ Python 3.8+
  ✓ CPU suffisant (pas besoin de GPU)
  ✓ ~500 MB RAM
  ✓ Micro intégré ou USB

═══════════════════════════════════════════════════════════════════════════

Version: 2.0 | Date: 8 janvier 2026 | Status: Production-ready
Compatible: Windows, Linux, macOS, Raspberry Pi

═══════════════════════════════════════════════════════════════════════════
