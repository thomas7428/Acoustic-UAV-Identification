#!/bin/bash
################################################################################
# Download Full DADS Dataset Offline
# 
# This script downloads the complete DADS dataset from Hugging Face
# and stores it in dataset_DADS_offline/ for offline use.
#
# The offline dataset will be automatically detected and used by:
# - download_and_prepare_dads.py
# - master_setup_v2.py
# - augment_dataset_v2.py
#
# This avoids repeated downloads when recreating augmented datasets.
#
# Usage:
#   ./download_full_dads_offline.sh
#
################################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/dataset_DADS_offline"
VENV_PATH="$SCRIPT_DIR/../.venv/bin/python"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        DOWNLOAD FULL DADS DATASET (OFFLINE MODE)          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if already exists
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null | wc -l)" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Warning: dataset_DADS_offline/ already exists and is not empty${NC}"
    echo ""
    echo "Current contents:"
    du -sh "$OUTPUT_DIR"/* 2>/dev/null || echo "  (empty or single file)"
    echo ""
    read -p "Do you want to re-download and overwrite? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Download cancelled${NC}"
        exit 0
    fi
    echo -e "${YELLOW}Cleaning existing dataset...${NC}"
    rm -rf "$OUTPUT_DIR"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}Output directory: ${BLUE}$OUTPUT_DIR${NC}"
echo ""

# Check if virtual environment exists
if [ ! -f "$VENV_PATH" ]; then
    echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
    echo "Please run setup first or adjust VENV_PATH"
    exit 1
fi

echo -e "${YELLOW}Starting full DADS dataset download...${NC}"
echo -e "${YELLOW}This will download ~180,000 audio files (several GB)${NC}"
echo -e "${YELLOW}Estimated time: 10-30 minutes depending on connection${NC}"
echo ""

# Run the download script without any limits (full dataset)
"$VENV_PATH" download_and_prepare_dads.py \
    --output_dir "$OUTPUT_DIR" \
    --verbose

# Check if download was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║          FULL DADS DATASET DOWNLOADED SUCCESSFULLY         ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}Dataset location: ${BLUE}$OUTPUT_DIR${NC}"
    echo ""
    echo "Dataset statistics:"
    echo "  Total size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
    echo "  Class 0 (ambient): $(find "$OUTPUT_DIR/0" -name "*.wav" 2>/dev/null | wc -l) files"
    echo "  Class 1 (drone): $(find "$OUTPUT_DIR/1" -name "*.wav" 2>/dev/null | wc -l) files"
    echo ""
    echo -e "${GREEN}✓ The offline dataset will be automatically used by all scripts${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                  DOWNLOAD FAILED                           ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${RED}Error occurred during download${NC}"
    echo "Please check:"
    echo "  - Internet connection"
    echo "  - Hugging Face availability"
    echo "  - Disk space"
    echo ""
    exit 1
fi
