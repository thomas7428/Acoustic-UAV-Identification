#!/bin/bash

# Test runner script for Acoustic UAV Identification project
# Runs all test suites with proper environment setup

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       🧪 ACOUSTIC UAV IDENTIFICATION - TEST SUITE             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Parse command-line arguments
TEST_MODE="all"
COVERAGE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_MODE="unit"
            shift
            ;;
        --integration)
            TEST_MODE="integration"
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: ./run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --unit          Run only unit tests"
            echo "  --integration   Run only integration tests"
            echo "  --coverage      Generate coverage report"
            echo "  --verbose       Verbose output"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${BLUE}🐍 Python version: ${PYTHON_VERSION}${NC}"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo -e "${YELLOW}📦 Activating virtual environment...${NC}"
    source "$PROJECT_ROOT/venv/bin/activate"
elif [ -d "$PROJECT_ROOT/.venv" ]; then
    echo -e "${YELLOW}📦 Activating virtual environment...${NC}"
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo -e "${YELLOW}⚠️  No virtual environment found. Using system Python.${NC}"
fi

# Check for required packages
echo -e "${BLUE}📋 Checking dependencies...${NC}"
MISSING_PACKAGES=()

python3 -c "import numpy" 2>/dev/null || MISSING_PACKAGES+=("numpy")
python3 -c "import tensorflow" 2>/dev/null || MISSING_PACKAGES+=("tensorflow")
python3 -c "import librosa" 2>/dev/null || MISSING_PACKAGES+=("librosa")

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${RED}❌ Missing required packages: ${MISSING_PACKAGES[*]}${NC}"
    echo -e "${YELLOW}💡 Install with: pip install ${MISSING_PACKAGES[*]}${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All dependencies available${NC}\n"

# Determine test runner
if command -v pytest &> /dev/null; then
    TEST_RUNNER="pytest"
    echo -e "${GREEN}✅ Using pytest${NC}"
else
    TEST_RUNNER="python3 -m unittest"
    echo -e "${YELLOW}⚠️  pytest not found, using unittest${NC}"
fi

# Run tests
cd "$SCRIPT_DIR"

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  RUNNING TESTS: ${TEST_MODE}${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

FAILED=0

if [ "$TEST_RUNNER" = "pytest" ]; then
    # Pytest command
    PYTEST_ARGS="-v"
    
    if [ "$COVERAGE" = true ]; then
        PYTEST_ARGS="$PYTEST_ARGS --cov --cov-report=html --cov-report=term"
    fi
    
    if [ "$VERBOSE" = true ]; then
        PYTEST_ARGS="$PYTEST_ARGS -vv"
    fi
    
    case $TEST_MODE in
        unit)
            PYTEST_ARGS="$PYTEST_ARGS -m unit"
            ;;
        integration)
            PYTEST_ARGS="$PYTEST_ARGS -m integration"
            ;;
    esac
    
    if pytest $PYTEST_ARGS; then
        echo -e "\n${GREEN}✅ All tests passed!${NC}"
    else
        FAILED=1
        echo -e "\n${RED}❌ Some tests failed${NC}"
    fi
    
else
    # Unittest command - run each test file individually
    TEST_FILES=(
        "test_augmentation.py"
        "test_loss_functions.py"
        "test_spec_augment.py"
    )
    
    for test_file in "${TEST_FILES[@]}"; do
        if [ -f "$test_file" ]; then
            echo -e "${BLUE}Running $test_file...${NC}"
            if python3 "$test_file"; then
                echo -e "${GREEN}✅ $test_file passed${NC}\n"
            else
                FAILED=1
                echo -e "${RED}❌ $test_file failed${NC}\n"
            fi
        fi
    done
    
    if [ $FAILED -eq 0 ]; then
        echo -e "\n${GREEN}✅ All tests passed!${NC}"
    else
        echo -e "\n${RED}❌ Some tests failed${NC}"
    fi
fi

# Display coverage report location if generated
if [ "$COVERAGE" = true ] && [ -d "htmlcov" ]; then
    echo ""
    echo -e "${BLUE}📊 Coverage report generated at: ${SCRIPT_DIR}/htmlcov/index.html${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  TEST RUN COMPLETE${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

exit $FAILED
