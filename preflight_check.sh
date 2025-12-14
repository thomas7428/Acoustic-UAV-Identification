#!/bin/bash

# Pre-flight checks before running full pipeline
# Verifies all dependencies, files, and configurations

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          🔍 PRE-FLIGHT CHECKS - PHASE 1 PIPELINE              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

CHECKS_PASSED=0
CHECKS_FAILED=0

# Function to check and report
check() {
    local name="$1"
    local command="$2"
    
    echo -n "🔍 Checking $name... "
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ OK${NC}"
        ((CHECKS_PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        ((CHECKS_FAILED++))
        return 1
    fi
}

# Activate virtual environment if it exists
PROJECT_ROOT="$(pwd)"
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo -e "${YELLOW}📦 Activating virtual environment (.venv)...${NC}"
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    echo -e "${YELLOW}📦 Activating virtual environment (venv)...${NC}"
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo -e "${YELLOW}⚠️  No virtual environment found. Using system Python.${NC}"
fi
echo ""

# 1. Python and dependencies
echo -e "${BLUE}═══ Python Environment ═══${NC}"
check "Python 3" "python3 --version"
check "NumPy" "python3 -c 'import numpy'"
check "TensorFlow" "python3 -c 'import tensorflow'"
check "Librosa" "python3 -c 'import librosa'"
check "SciPy" "python3 -c 'import scipy'"
echo ""

# 2. Critical files
echo -e "${BLUE}═══ Critical Files ═══${NC}"
check "augment_dataset_v2.py" "[ -f '0 - DADS dataset extraction/augment_dataset_v2.py' ]"
check "augment_config_v2.json" "[ -f '0 - DADS dataset extraction/augment_config_v2.json' ]"
check "loss_functions.py" "[ -f '2 - Model Training/loss_functions.py' ]"
check "CNN_Trainer.py" "[ -f '2 - Model Training/CNN_Trainer.py' ]"
check "RNN_Trainer.py" "[ -f '2 - Model Training/RNN_Trainer.py' ]"
check "CRNN_Trainer.py" "[ -f '2 - Model Training/CRNN_Trainer.py' ]"
check "Mel_Preprocess_and_Feature_Extract.py" "[ -f '1 - Preprocessing and Features Extraction/Mel_Preprocess_and_Feature_Extract.py' ]"
echo ""

# 3. Test framework
echo -e "${BLUE}═══ Test Framework ═══${NC}"
check "test_augmentation.py" "[ -f '7 - Tests/test_augmentation.py' ]"
check "test_loss_functions.py" "[ -f '7 - Tests/test_loss_functions.py' ]"
check "test_spec_augment.py" "[ -f '7 - Tests/test_spec_augment.py' ]"
check "run_tests.sh" "[ -x '7 - Tests/run_tests.sh' ]"
echo ""

# 4. Pipeline script
echo -e "${BLUE}═══ Pipeline Script ═══${NC}"
check "run_full_pipeline.sh" "[ -x 'run_full_pipeline.sh' ]"
echo ""

# 5. Dataset structure
echo -e "${BLUE}═══ Dataset Structure ═══${NC}"
check "dataset_train/0/" "[ -d '0 - DADS dataset extraction/dataset_train/0' ]"
check "dataset_train/1/" "[ -d '0 - DADS dataset extraction/dataset_train/1' ]"
check "dataset_test/0/" "[ -d '0 - DADS dataset extraction/dataset_test/0' ]"
check "dataset_test/1/" "[ -d '0 - DADS dataset extraction/dataset_test/1' ]"
echo ""

# 6. Verify augmentation functions exist
echo -e "${BLUE}═══ Augmentation Functions ═══${NC}"
if grep -q "def apply_doppler_shift" "0 - DADS dataset extraction/augment_dataset_v2.py"; then
    echo -e "🔍 Checking apply_doppler_shift... ${GREEN}✅ OK${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "🔍 Checking apply_doppler_shift... ${RED}❌ FAIL${NC}"
    ((CHECKS_FAILED++))
fi

if grep -q "def apply_intensity_modulation" "0 - DADS dataset extraction/augment_dataset_v2.py"; then
    echo -e "🔍 Checking apply_intensity_modulation... ${GREEN}✅ OK${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "🔍 Checking apply_intensity_modulation... ${RED}❌ FAIL${NC}"
    ((CHECKS_FAILED++))
fi

if grep -q "def apply_reverberation" "0 - DADS dataset extraction/augment_dataset_v2.py"; then
    echo -e "🔍 Checking apply_reverberation... ${GREEN}✅ OK${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "🔍 Checking apply_reverberation... ${RED}❌ FAIL${NC}"
    ((CHECKS_FAILED++))
fi

if grep -q "def apply_time_stretch_variation" "0 - DADS dataset extraction/augment_dataset_v2.py"; then
    echo -e "🔍 Checking apply_time_stretch_variation... ${GREEN}✅ OK${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "🔍 Checking apply_time_stretch_variation... ${RED}❌ FAIL${NC}"
    ((CHECKS_FAILED++))
fi
echo ""

# 7. Verify loss functions
echo -e "${BLUE}═══ Loss Functions ═══${NC}"
if grep -q "def focal_loss" "2 - Model Training/loss_functions.py"; then
    echo -e "🔍 Checking focal_loss... ${GREEN}✅ OK${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "🔍 Checking focal_loss... ${RED}❌ FAIL${NC}"
    ((CHECKS_FAILED++))
fi

if grep -q "def weighted_binary_crossentropy" "2 - Model Training/loss_functions.py"; then
    echo -e "🔍 Checking weighted_binary_crossentropy... ${GREEN}✅ OK${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "🔍 Checking weighted_binary_crossentropy... ${RED}❌ FAIL${NC}"
    ((CHECKS_FAILED++))
fi

if grep -q "def get_loss_function" "2 - Model Training/loss_functions.py"; then
    echo -e "🔍 Checking get_loss_function... ${GREEN}✅ OK${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "🔍 Checking get_loss_function... ${RED}❌ FAIL${NC}"
    ((CHECKS_FAILED++))
fi
echo ""

# 8. Verify SpecAugment
echo -e "${BLUE}═══ SpecAugment ═══${NC}"
if grep -q "def spec_augment" "1 - Preprocessing and Features Extraction/Mel_Preprocess_and_Feature_Extract.py"; then
    echo -e "🔍 Checking spec_augment function... ${GREEN}✅ OK${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "🔍 Checking spec_augment function... ${RED}❌ FAIL${NC}"
    ((CHECKS_FAILED++))
fi
echo ""

# 9. Quick syntax check
echo -e "${BLUE}═══ Python Syntax Checks ═══${NC}"
check "augment_dataset_v2.py syntax" "python3 -m py_compile '0 - DADS dataset extraction/augment_dataset_v2.py'"
check "loss_functions.py syntax" "python3 -m py_compile '2 - Model Training/loss_functions.py'"
check "CNN_Trainer.py syntax" "python3 -m py_compile '2 - Model Training/CNN_Trainer.py'"
check "Mel_Preprocess_and_Feature_Extract.py syntax" "python3 -m py_compile '1 - Preprocessing and Features Extraction/Mel_Preprocess_and_Feature_Extract.py'"
echo ""

# 10. Run quick unit tests
echo -e "${BLUE}═══ Quick Unit Test Sample ═══${NC}"
echo "Running SpecAugment tests (quick sample)..."
if cd "7 - Tests" && python3 test_spec_augment.py > /dev/null 2>&1; then
    echo -e "🔍 SpecAugment tests... ${GREEN}✅ All 17 tests passed${NC}"
    ((CHECKS_PASSED++))
    cd ..
else
    echo -e "🔍 SpecAugment tests... ${RED}❌ Some tests failed${NC}"
    ((CHECKS_FAILED++))
    cd ..
fi
echo ""

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                      📊 CHECK SUMMARY                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Total checks: $((CHECKS_PASSED + CHECKS_FAILED))"
echo -e "${GREEN}✅ Passed: $CHECKS_PASSED${NC}"
echo -e "${RED}❌ Failed: $CHECKS_FAILED${NC}"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✅ ALL CHECKS PASSED - READY TO RUN PIPELINE                 ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}📋 Next steps:${NC}"
    echo -e "  1. Run full pipeline: ${YELLOW}./run_full_pipeline.sh --parallel${NC}"
    echo -e "  2. Monitor logs: ${YELLOW}tail -f logs/pipeline_*.log${NC}"
    echo -e "  3. Expected duration: ~45 minutes with parallel training"
    echo -e "  4. Results will be in: ${YELLOW}results/${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ❌ SOME CHECKS FAILED - PLEASE FIX BEFORE RUNNING            ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}💡 Troubleshooting tips:${NC}"
    echo -e "  - Install missing Python packages: pip install <package>"
    echo -e "  - Check file permissions: chmod +x <script>"
    echo -e "  - Verify dataset structure in '0 - DADS dataset extraction/'"
    echo -e "  - Review error messages above"
    echo ""
    exit 1
fi
