@echo off
REM ============================================================================
REM Test rapide du systeme de detection - Windows
REM ============================================================================

echo ========================================================================
echo   TEST SYSTEME DE DETECTION DE DRONES (Windows)
echo ========================================================================
echo.

REM Test 1: Dependances
echo [1/3] Verification des dependances Python...
python -c "import tensorflow, librosa, numpy, soundfile" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] Dependances manquantes!
    echo.
    echo Installation requise:
    echo   pip install tensorflow librosa numpy soundfile
    echo.
    pause
    exit /b 1
)
echo OK - Dependances installees

REM Test 2: Modeles
echo [2/3] Verification des modeles...
if not exist "models\*.keras" (
    echo [ERREUR] Modeles manquants dans models\
    pause
    exit /b 1
)
echo OK - Modeles presents

REM Test 3: Detection
echo [3/3] Test de detection...
echo.

if exist "audio_input\test_drone.wav" (
    echo Test avec fichier drone...
    python drone_detector.py --file audio_input\test_drone.wav 2>nul | findstr /C:"DRONE" /C:"ALERT" /C:"Confidence"
    echo.
) else (
    echo [WARN] Pas de fichier test_drone.wav
)

echo ========================================================================
echo   RESULTAT DU TEST
echo ========================================================================
echo.
echo Si vous voyez "ALERT" et "DRONE" ci-dessus, le systeme fonctionne!
echo.
echo Prochaines etapes:
echo   1. Test avec votre micro:
echo      start_detection.bat --with-recording
echo.
echo   2. Voir le guide complet:
echo      WINDOWS_GUIDE.md
echo.
pause
