#!/bin/bash
################################################################################
# Quick Start Script for Deployment
# Run this after copying to Raspberry Pi
################################################################################

set -e

echo "========================================================================"
echo "  RASPBERRY PI DRONE DETECTION - QUICK START"
echo "========================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check if running on Raspberry Pi
if [ -f /proc/device-tree/model ]; then
    PI_MODEL=$(cat /proc/device-tree/model)
    echo -e "${CYAN}Detected:${NC} $PI_MODEL"
else
    echo -e "${YELLOW}Warning:${NC} Not running on Raspberry Pi (or model not detected)"
fi

echo ""
echo "This script will:"
echo "  1. Check system dependencies"
echo "  2. Install Python packages"
echo "  3. Test the deployment"
echo "  4. Provide next steps"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

echo ""
echo "========================================================================"
echo "  STEP 1: System Dependencies"
echo "========================================================================"

# Update package list
echo -e "${CYAN}→${NC} Updating package list..."
sudo apt-get update -qq

# Install system dependencies
echo -e "${CYAN}→${NC} Installing system dependencies..."
sudo apt-get install -y python3-pip portaudio19-dev libatlas-base-dev

echo -e "${GREEN}✓${NC} System dependencies installed"

echo ""
echo "========================================================================"
echo "  STEP 2: Python Packages"
echo "========================================================================"

# Check if we should use TensorFlow Lite (recommended for Pi)
echo ""
echo "TensorFlow options:"
echo "  1. TensorFlow Lite (recommended, faster on Pi)"
echo "  2. Full TensorFlow (slower, more features)"
echo ""
read -p "Choose option (1 or 2): " -n 1 -r TF_CHOICE
echo ""

if [[ $TF_CHOICE == "1" ]]; then
    echo -e "${CYAN}→${NC} Installing TensorFlow Lite..."
    pip3 install --upgrade pip
    pip3 install tflite-runtime
    echo -e "${GREEN}✓${NC} TensorFlow Lite installed"
else
    echo -e "${CYAN}→${NC} Installing TensorFlow (this may take a while)..."
    pip3 install --upgrade pip
    pip3 install tensorflow
    echo -e "${GREEN}✓${NC} TensorFlow installed"
fi

# Install other packages
echo -e "${CYAN}→${NC} Installing audio processing libraries..."
pip3 install librosa soundfile numpy

# Optional: PyAudio for recording
read -p "Install PyAudio for microphone recording? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo apt-get install -y python3-pyaudio
    echo -e "${GREEN}✓${NC} PyAudio installed"
fi

echo ""
echo "========================================================================"
echo "  STEP 3: Testing Deployment"
echo "========================================================================"

python3 test_deployment.py

echo ""
echo "========================================================================"
echo "  SETUP COMPLETE!"
echo "========================================================================"
echo ""
echo -e "${GREEN}✓ Deployment ready!${NC}"
echo ""
echo "Next steps:"
echo ""
echo -e "${CYAN}1. Test with a sample file:${NC}"
echo "   cp /path/to/audio.wav audio_input/"
echo "   python3 drone_detector.py --file audio_input/audio.wav"
echo ""
echo -e "${CYAN}2. Start continuous monitoring:${NC}"
echo "   python3 drone_detector.py --continuous"
echo ""
echo -e "${CYAN}3. Record and detect:${NC}"
echo "   # Terminal 1:"
echo "   python3 audio_recorder.py --interval 5"
echo "   # Terminal 2:"
echo "   python3 drone_detector.py --continuous"
echo ""
echo -e "${CYAN}4. Run as background service:${NC}"
echo "   nohup python3 drone_detector.py --continuous > logs/detector.log 2>&1 &"
echo ""
echo "Configuration:"
echo "  • Edit thresholds: vim deployment_config.json"
echo "  • View logs: tail -f logs/detector_$(date +%Y%m%d).log"
echo "  • Check predictions: cat logs/predictions.json | jq"
echo ""
echo "========================================================================"
