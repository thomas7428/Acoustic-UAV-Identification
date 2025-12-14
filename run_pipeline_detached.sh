#!/bin/bash
################################################################################
# Acoustic UAV Detection - Detached Pipeline Runner
# 
# This script runs the full pipeline in a way that survives SSH disconnections.
# It uses nohup to detach the process and provides real-time log monitoring.
#
# Usage:
#   ./run_pipeline_detached.sh [OPTIONS]
#
# All options are passed directly to run_full_pipeline.sh
#
# Examples:
#   ./run_pipeline_detached.sh --skip-dataset --parallel
#   ./run_pipeline_detached.sh --parallel --no-visualizations
#
# Monitoring:
#   tail -f logs/pipeline_YYYYMMDD_HHMMSS.log
#   tail -f logs/*_training_*.log
################################################################################

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_DIR/logs"
NOHUP_LOG="$LOG_DIR/nohup_${TIMESTAMP}.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

mkdir -p "$LOG_DIR"

echo -e "${CYAN}================================${NC}"
echo -e "${CYAN}  Detached Pipeline Runner${NC}"
echo -e "${CYAN}================================${NC}"
echo ""
echo -e "${GREEN}Starting pipeline in detached mode...${NC}"
echo -e "${GREEN}Process will continue even if SSH connection is lost.${NC}"
echo ""
echo -e "${YELLOW}Arguments passed:${NC} $@"
echo -e "${YELLOW}Nohup log:${NC} $NOHUP_LOG"
echo ""

# Run pipeline with nohup and disown
cd "$PROJECT_DIR"
nohup bash run_full_pipeline.sh "$@" > "$NOHUP_LOG" 2>&1 &
PIPELINE_PID=$!

# Disown the process so it survives shell exit
disown

echo -e "${GREEN}✓ Pipeline started successfully!${NC}"
echo ""
echo -e "${CYAN}Process ID:${NC} $PIPELINE_PID"
echo -e "${CYAN}Nohup log:${NC} $NOHUP_LOG"
echo ""
echo -e "${YELLOW}Monitor progress with:${NC}"
echo "  tail -f $NOHUP_LOG"
echo ""
echo -e "${YELLOW}Or check specific training logs:${NC}"
echo "  tail -f $LOG_DIR/cnn_training_*.log"
echo "  tail -f $LOG_DIR/rnn_training_*.log"
echo "  tail -f $LOG_DIR/crnn_training_*.log"
echo "  tail -f $LOG_DIR/multitask_cnn_training_*.log"
echo ""
echo -e "${YELLOW}Check if still running:${NC}"
echo "  ps aux | grep $PIPELINE_PID"
echo "  ps aux | grep 'CNN_Trainer\\|RNN_Trainer\\|CRNN_Trainer\\|MultiTask_CNN'"
echo ""
echo -e "${YELLOW}Kill pipeline if needed:${NC}"
echo "  kill $PIPELINE_PID"
echo "  pkill -f 'CNN_Trainer\\|RNN_Trainer\\|CRNN_Trainer\\|MultiTask_CNN'"
echo ""
echo -e "${GREEN}You can now safely disconnect from SSH.${NC}"
echo -e "${GREEN}The pipeline will continue running in the background.${NC}"
