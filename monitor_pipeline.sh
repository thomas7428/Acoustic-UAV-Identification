#!/bin/bash
################################################################################
# Complete Pipeline Monitoring Script (Extended from monitor_training.sh)
# Monitors full pipeline execution with dataset, training, and analysis stages
# Usage: bash monitor_pipeline.sh [log_file]
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get latest log file (try Phase 2F first, then any phase log)
if [ -z "$1" ]; then
    LOG_FILE=$(ls -t logs/phase2f_fullpipeline_*.log logs/phase2f_v3_*.log logs/phase*.log 2>/dev/null | head -1)
    if [ -z "$LOG_FILE" ]; then
        echo -e "${RED}[ERROR] No pipeline log file found${NC}"
        echo "Usage: bash monitor_pipeline.sh [log_file]"
        exit 1
    fi
else
    LOG_FILE="$1"
fi

echo -e "${CYAN}=================================${NC}"
echo -e "${CYAN} COMPLETE PIPELINE MONITOR${NC}"
echo -e "${CYAN}=================================${NC}"
echo -e "Log file: ${YELLOW}$LOG_FILE${NC}"
echo -e "${CYAN}=================================${NC}\n"

# Check if pipeline is running
PID=$(pgrep -f "run_full_pipeline.sh.*augment_config_v3")
if [ -z "$PID" ]; then
    echo -e "${YELLOW}[WARNING] Pipeline process not found (may have finished)${NC}\n"
else
    echo -e "${GREEN}[RUNNING] Pipeline PID: $PID${NC}\n"
fi

# Function to extract and display current step
show_current_step() {
    echo -e "${MAGENTA}[CURRENT STEP]${NC}"
    grep -E "\[STEP\]|\[INFO\] (Training|Calculating|Running|Generating)" "$LOG_FILE" | tail -5
    echo ""
}

# Function to show progress
show_progress() {
    echo -e "${BLUE}[PROGRESS SUMMARY]${NC}"
    
    # Dataset generation
    if grep -q "Dataset generation completed\|Dataset pipeline completed\|Setup complete.*datasets are ready" "$LOG_FILE"; then
        echo -e "  ${GREEN}✓${NC} Step 1: Dataset Generation"
    elif grep -q "STEP 1/5\|Generating.*dataset\|Augmenting" "$LOG_FILE"; then
        echo -e "  ${YELLOW}⏳${NC} Step 1: Dataset Generation (in progress)"
    else
        echo -e "  ${NC}○${NC} Step 1: Dataset Generation"
    fi
    
    # Feature extraction
    if grep -q "Feature extraction complete\|MEL feature extraction completed\|MFCC feature extraction completed\|EXTRACTION COMPLETE" "$LOG_FILE"; then
        echo -e "  ${GREEN}✓${NC} Step 2: Feature Extraction"
    elif grep -q "STEP 2/5\|STEP 1.5/5: Feature Extraction\|Extracting.*features" "$LOG_FILE"; then
        echo -e "  ${YELLOW}⏳${NC} Step 2: Feature Extraction (in progress)"
    else
        echo -e "  ${NC}○${NC} Step 2: Feature Extraction"
    fi
    
    # Model training
    if grep -q "CNN training completed\|\[✓\] CNN training completed" "$LOG_FILE"; then
        echo -e "  ${GREEN}✓${NC} Step 3a: CNN Training"
    elif grep -q "Training CNN\|CNN_Trainer.py" "$LOG_FILE"; then
        echo -e "  ${YELLOW}⏳${NC} Step 3a: CNN Training (in progress)"
    else
        echo -e "  ${NC}○${NC} Step 3a: CNN Training"
    fi
    
    if grep -q "RNN training completed\|\[✓\] RNN training completed" "$LOG_FILE"; then
        echo -e "  ${GREEN}✓${NC} Step 3b: RNN Training"
    elif grep -q "Training RNN\|RNN_Trainer.py" "$LOG_FILE"; then
        echo -e "  ${YELLOW}⏳${NC} Step 3b: RNN Training (in progress)"
    else
        echo -e "  ${NC}○${NC} Step 3b: RNN Training"
    fi
    
    if grep -q "CRNN training completed\|\[✓\] CRNN training completed" "$LOG_FILE"; then
        echo -e "  ${GREEN}✓${NC} Step 3c: CRNN Training"
    elif grep -q "Training CRNN\|CRNN_Trainer.py" "$LOG_FILE"; then
        echo -e "  ${YELLOW}⏳${NC} Step 3c: CRNN Training (in progress)"
    else
        echo -e "  ${NC}○${NC} Step 3c: CRNN Training"
    fi
    
    if grep -q "Attention-Enhanced CRNN training completed\|Attention_CRNN training completed" "$LOG_FILE"; then
        echo -e "  ${GREEN}✓${NC} Step 3d: Attention-CRNN Training"
    elif grep -q "Training Attention-Enhanced CRNN\|Training Attention_CRNN" "$LOG_FILE"; then
        echo -e "  ${YELLOW}⏳${NC} Step 3d: Attention-CRNN Training (in progress)"
    else
        echo -e "  ${NC}○${NC} Step 3d: Attention-CRNN Training"
    fi
    
    # Performance calculations
    if grep -q "performance calculated" "$LOG_FILE" | tail -1 | grep -q "performance calculated"; then
        echo -e "  ${GREEN}✓${NC} Step 4: Performance Calculations"
    elif grep -q "STEP 3/5" "$LOG_FILE"; then
        echo -e "  ${YELLOW}⏳${NC} Step 4: Performance Calculations (in progress)"
    else
        echo -e "  ${NC}○${NC} Step 4: Performance Calculations"
    fi
    
    # Threshold calibration
    if grep -q "Threshold calibration completed" "$LOG_FILE"; then
        echo -e "  ${GREEN}✓${NC} Step 5: Threshold Calibration"
    elif grep -q "STEP 4/6" "$LOG_FILE"; then
        echo -e "  ${YELLOW}⏳${NC} Step 5: Threshold Calibration (in progress)"
    else
        echo -e "  ${NC}○${NC} Step 5: Threshold Calibration"
    fi
    
    # Visualizations
    if grep -q "visualizations generated" "$LOG_FILE"; then
        echo -e "  ${GREEN}✓${NC} Step 6: Visualizations"
    elif grep -q "STEP 5/6" "$LOG_FILE"; then
        echo -e "  ${YELLOW}⏳${NC} Step 6: Visualizations (in progress)"
    else
        echo -e "  ${NC}○${NC} Step 6: Visualizations"
    fi
    
    echo ""
}

# Function to show errors/warnings
show_issues() {
    ERROR_COUNT=$(grep -c "\[ERROR\]" "$LOG_FILE" 2>/dev/null | head -1)
    WARN_COUNT=$(grep -c "\[WARN\]" "$LOG_FILE" 2>/dev/null | head -1)
    
    # Ensure we have valid integers
    ERROR_COUNT=${ERROR_COUNT:-0}
    WARN_COUNT=${WARN_COUNT:-0}
    
    if [ "$ERROR_COUNT" -gt 0 ] 2>/dev/null || [ "$WARN_COUNT" -gt 0 ] 2>/dev/null; then
        echo -e "${RED}[ISSUES DETECTED]${NC}"
        echo -e "  Errors: ${RED}$ERROR_COUNT${NC}"
        echo -e "  Warnings: ${YELLOW}$WARN_COUNT${NC}"
        
        if [ "$ERROR_COUNT" -gt 0 ] 2>/dev/null; then
            echo -e "\n${RED}Recent errors:${NC}"
            grep "\[ERROR\]" "$LOG_FILE" | tail -3
        fi
        echo ""
    fi
}

# Function to get training progress from individual training logs
get_training_progress() {
    local model=$1
    local log_file="logs/${model}_training_*.log"
    
    # Find the most recent log file
    local latest_log=$(ls -t $log_file 2>/dev/null | head -1)
    
    if [ -f "$latest_log" ]; then
        # Find the last line with COMPLETE epoch results
        local summary_line=$(grep -E "step - accuracy:.*val_accuracy:.*val_loss:" "$latest_log" | tail -1)
        
        if [ -n "$summary_line" ]; then
            # Extract epoch number
            local line_num=$(grep -n "step - accuracy:.*val_accuracy:.*val_loss:" "$latest_log" | tail -1 | cut -d: -f1)
            local epoch=$(head -n $line_num "$latest_log" | grep -E "^Epoch [0-9]+/" | tail -1 | grep -oP "Epoch \K[0-9]+(?=/)")
            
            # Extract validation accuracy and loss
            local val_acc=$(echo "$summary_line" | grep -oP "val_accuracy: \K[0-9.]+")
            local val_loss=$(echo "$summary_line" | grep -oP "val_loss: \K[0-9.]+")
            
            if [ -n "$val_acc" ] && [ -n "$val_loss" ]; then
                # Convert to percentage and format
                local val_acc_pct=$(echo "$val_acc * 100" | LC_NUMERIC=C bc 2>/dev/null | LC_NUMERIC=C awk '{printf "%.2f", $1}')
                local val_loss_fmt=$(echo "$val_loss" | LC_NUMERIC=C awk '{printf "%.4f", $1}')
                
                echo -e "    ${GREEN}$model${NC}: Epoch ${YELLOW}$epoch${NC} | Val Acc: ${CYAN}${val_acc_pct}%${NC} | Val Loss: ${CYAN}${val_loss_fmt}${NC}"
            else
                echo -e "    ${YELLOW}$model${NC}: Epoch ${epoch} (in progress...)"
            fi
        else
            # No complete epoch yet
            if grep -q "^Epoch 1/" "$latest_log"; then
                echo -e "    ${YELLOW}$model${NC}: Epoch 1 (in progress...)"
            else
                echo -e "    ${YELLOW}$model${NC}: Starting..."
            fi
        fi
    else
        echo -e "    ${NC}$model${NC}: Not started"
    fi
}

# Function to show training metrics
show_training_metrics() {
    # Check if any training has started
    if ls logs/*_training_*.log 1> /dev/null 2>&1; then
        echo -e "${GREEN}[TRAINING METRICS - LIVE]${NC}"
        
        get_training_progress "cnn"
        get_training_progress "rnn"
        get_training_progress "crnn"
        get_training_progress "attention_crnn"
        
        echo ""
    fi
}

# Main monitoring loop
export LC_NUMERIC=C
echo -e "${CYAN}Press Ctrl+C to exit monitoring${NC}\n"
echo -e "${CYAN}=================================${NC}\n"

while true; do
    clear
    
    echo -e "${CYAN}=================================${NC}"
    echo -e "${CYAN} COMPLETE PIPELINE MONITOR${NC}"
    echo -e "${CYAN}=================================${NC}"
    echo -e "Log: ${YELLOW}$(basename $LOG_FILE)${NC}"
    echo -e "Time: ${YELLOW}$(date '+%Y-%m-%d %H:%M:%S')${NC}"
    
    # Check if still running (support multiple pipeline scripts)
    PID=$(pgrep -f "run_full_pipeline.sh\|master_setup")
    if [ -z "$PID" ]; then
        echo -e "Status: ${YELLOW}FINISHED or STOPPED${NC}"
    else
        echo -e "Status: ${GREEN}RUNNING (PID: $PID)${NC}"
    fi
    
    # Check for active trainers
    TRAINERS=$(ps aux | grep -E "(CNN|RNN|CRNN|Attention_CRNN)_Trainer.py" | grep -v grep | wc -l)
    if [ "$TRAINERS" -gt 0 ]; then
        echo -e "Active Trainers: ${GREEN}$TRAINERS${NC}"
    fi
    
    echo -e "${CYAN}=================================${NC}\n"
    
    show_current_step
    show_progress
    show_issues
    show_training_metrics
    
    # Check if pipeline finished
    if grep -q "Pipeline completed\|PIPELINE COMPLETED" "$LOG_FILE"; then
        echo -e "${GREEN}=================================${NC}"
        echo -e "${GREEN} PIPELINE COMPLETED!${NC}"
        echo -e "${GREEN}=================================${NC}"
        
        # Show final summary
        DURATION=$(grep "Total Duration\|Duration:" "$LOG_FILE" | tail -1)
        if [ ! -z "$DURATION" ]; then
            echo -e "\n$DURATION"
        fi
        
        echo -e "\n${CYAN}Check results:${NC}"
        echo -e "  • Visualizations: ${YELLOW}6 - Visualization/outputs/${NC}"
        echo -e "  • Performance: ${YELLOW}0 - DADS dataset extraction/results/${NC}"
        echo -e "  • Models: ${YELLOW}0 - DADS dataset extraction/saved_models/${NC}"
        
        break
    fi
    
    # Show last log lines (more context)
    echo -e "${BLUE}[RECENT ACTIVITY]${NC}"
    tail -5 "$LOG_FILE" | grep -v "^$"
    echo ""
    
    sleep 5
done

echo -e "\n${CYAN}Monitoring stopped.${NC}"
echo -e "Full log: ${YELLOW}$LOG_FILE${NC}"
