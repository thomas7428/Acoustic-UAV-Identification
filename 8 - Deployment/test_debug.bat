@echo off
setlocal enabledelayedexpansion

set MODE=detection_only

echo Testing MODE value: %MODE%

if "%MODE%"=="detection_only" (
    echo ========================================================================
    echo   MODE DETECTION SEULE
    echo ========================================================================
    echo.
    echo Le systeme surveille le dossier: audio_input\
    echo.
    echo Pour ajouter des fichiers audio a analyser:
    echo   1. Copier fichiers WAV vers audio_input\
    echo   2. Analyse automatique toutes les 5 secondes
    echo.
    echo Appuyez sur Ctrl+C pour arreter
    echo.
)

echo Script complete
pause
