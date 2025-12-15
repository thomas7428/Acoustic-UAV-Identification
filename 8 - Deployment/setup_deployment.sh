#!/bin/bash
################################################################################
# Setup Script for Raspberry Pi Deployment
# Copies trained models to deployment directory
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================================================"
echo "  RASPBERRY PI DEPLOYMENT SETUP"
echo "========================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Paths
MODELS_SOURCE="$PROJECT_ROOT/0 - DADS dataset extraction/saved_models"
MODELS_DEST="$SCRIPT_DIR/models"

echo "Source models directory: $MODELS_SOURCE"
echo "Destination directory: $MODELS_DEST"
echo ""

# Check if source directory exists
if [ ! -d "$MODELS_SOURCE" ]; then
    echo -e "${RED}[ERROR]${NC} Models directory not found: $MODELS_SOURCE"
    echo "Please train the models first!"
    exit 1
fi

# Create destination directory
mkdir -p "$MODELS_DEST"

# Copy models
echo "Copying trained models..."
echo ""

MODELS_COPIED=0

# CNN Model
if [ -f "$MODELS_SOURCE/cnn_model.keras" ]; then
    cp "$MODELS_SOURCE/cnn_model.keras" "$MODELS_DEST/"
    echo -e "  ${GREEN}✓${NC} CNN model copied"
    MODELS_COPIED=$((MODELS_COPIED + 1))
else
    echo -e "  ${YELLOW}⊗${NC} CNN model not found (skipped)"
fi

# RNN Model
if [ -f "$MODELS_SOURCE/rnn_model.keras" ]; then
    cp "$MODELS_SOURCE/rnn_model.keras" "$MODELS_DEST/"
    echo -e "  ${GREEN}✓${NC} RNN model copied"
    MODELS_COPIED=$((MODELS_COPIED + 1))
else
    echo -e "  ${YELLOW}⊗${NC} RNN model not found (skipped)"
fi

# CRNN Model
if [ -f "$MODELS_SOURCE/crnn_model.keras" ]; then
    cp "$MODELS_SOURCE/crnn_model.keras" "$MODELS_DEST/"
    echo -e "  ${GREEN}✓${NC} CRNN model copied"
    MODELS_COPIED=$((MODELS_COPIED + 1))
else
    echo -e "  ${YELLOW}⊗${NC} CRNN model not found (skipped)"
fi

# Attention-CRNN Model
if [ -f "$MODELS_SOURCE/attention_crnn_model.keras" ]; then
    cp "$MODELS_SOURCE/attention_crnn_model.keras" "$MODELS_DEST/"
    echo -e "  ${GREEN}✓${NC} Attention-CRNN model copied"
    MODELS_COPIED=$((MODELS_COPIED + 1))
else
    echo -e "  ${YELLOW}⊗${NC} Attention-CRNN model not found (skipped)"
fi

echo ""
echo "========================================================================"
echo "  SETUP COMPLETE"
echo "========================================================================"
echo ""
echo "Models copied: $MODELS_COPIED"
echo "Deployment directory: $SCRIPT_DIR"
echo ""

if [ $MODELS_COPIED -eq 0 ]; then
    echo -e "${RED}[WARNING]${NC} No models were copied!"
    echo "Please ensure models are trained before deployment."
    exit 1
else
    echo -e "${GREEN}✓ Ready for deployment!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Transfer this directory to Raspberry Pi:"
    echo "     scp -r '8 - Deployment' pi@raspberrypi:/home/pi/"
    echo ""
    echo "  2. On Raspberry Pi, install dependencies:"
    echo "     pip install tensorflow librosa numpy"
    echo ""
    echo "  3. Run detector:"
    echo "     python3 drone_detector.py --continuous"
    echo ""
fi
