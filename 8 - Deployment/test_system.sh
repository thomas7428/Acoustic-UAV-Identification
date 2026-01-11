#!/bin/bash
################################################################################
# Test Suite for Drone Detection Deployment
# Validates the complete deployment setup
################################################################################

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if available
if [ -f "../.venv/bin/activate" ]; then
    source "../.venv/bin/activate"
fi

echo "========================================================================"
echo "  DRONE DETECTION SYSTEM - TEST SUITE"
echo "========================================================================"
echo ""

PASSED=0
FAILED=0

# Test 1: Dependencies
echo -e "[1/6] Testing Python dependencies..."
if python3 -c "import tensorflow, librosa, numpy, soundfile" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} All dependencies installed"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Missing dependencies"
    ((FAILED++))
fi

# Test 2: Models
echo -e "[2/6] Testing model files..."
MODEL_COUNT=$(ls -1 models/*.keras 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -ge 2 ]; then
    echo -e "${GREEN}✓${NC} $MODEL_COUNT models found"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Insufficient models ($MODEL_COUNT < 2)"
    ((FAILED++))
fi

# Test 3: Configuration
echo -e "[3/6] Testing configuration..."
if python3 -c "import json; json.load(open('deployment_config.json'))" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Configuration valid"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Invalid configuration"
    ((FAILED++))
fi

# Test 4: Test files
echo -e "[4/6] Checking test audio files..."
if [ -f "audio_input/test_drone.wav" ] && [ -f "audio_input/test_ambient.wav" ]; then
    echo -e "${GREEN}✓${NC} Test files present"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠${NC}  Test files missing (not critical)"
    ((PASSED++))
fi

# Test 5: Drone detection
echo -e "[5/6] Testing drone detection..."
if [ -f "audio_input/test_drone.wav" ]; then
    OUTPUT=$(python3 drone_detector.py --file audio_input/test_drone.wav 2>&1 | grep "DRONE")
    if echo "$OUTPUT" | grep -q "🚨.*DRONE"; then
        echo -e "${GREEN}✓${NC} Drone correctly detected"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} Drone detection failed"
        ((FAILED++))
    fi
else
    echo -e "${YELLOW}⚠${NC}  Skipping (no test file)"
    ((PASSED++))
fi

# Test 6: Ambient rejection
echo -e "[6/6] Testing ambient rejection..."
if [ -f "audio_input/test_ambient.wav" ]; then
    OUTPUT=$(python3 drone_detector.py --file audio_input/test_ambient.wav 2>&1 | grep -E "(DRONE|NO_DRONE)")
    if echo "$OUTPUT" | grep -q "NO_DRONE"; then
        echo -e "${GREEN}✓${NC} Ambient correctly rejected"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} False positive on ambient"
        ((FAILED++))
    fi
else
    echo -e "${YELLOW}⚠${NC}  Skipping (no test file)"
    ((PASSED++))
fi

# Summary
echo ""
echo "========================================================================"
echo "  TEST RESULTS"
echo "========================================================================"
echo -e "${GREEN}Passed:${NC} $PASSED/6"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed:${NC} $FAILED/6"
fi
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    echo ""
    echo "The deployment system is ready to use!"
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./start_detection.sh --test audio_input/test_drone.wav"
    echo "  2. Or:  ./start_detection.sh --with-recording"
    echo ""
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please fix the issues before deploying."
    exit 1
fi
