#!/bin/bash
# Training Monitoring Script
# Usage: bash monitor_training.sh [log_file]

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Find the most recent log file
if [ -z "$1" ]; then
    LOG_FILE=$(ls -t logs/phase*.log 2>/dev/null | head -1)
    if [ -z "$LOG_FILE" ]; then
        echo "No training log found. Please specify a log file."
        echo "Usage: bash monitor_training.sh [log_file]"
        exit 1
    fi
else
    LOG_FILE="$1"
fi

# Get timestamp from log file to match with correct history files
LOG_TIMESTAMP=$(grep "Timestamp:" "$LOG_FILE" 2>/dev/null | head -1 | awk '{print $3}')
RESULTS_DIR="0 - DADS dataset extraction/results"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  TRAINING PROGRESS MONITOR${NC}"
echo -e "${CYAN}========================================${NC}"
echo -e "Monitoring: ${YELLOW}$LOG_FILE${NC}"
if [ -n "$LOG_TIMESTAMP" ]; then
    echo -e "Training Session: ${YELLOW}$LOG_TIMESTAMP${NC}"
fi
echo ""

# Function to extract epoch info from training logs
get_training_progress() {
    local model=$1
    local log_file="logs/${model}_training_*.log"
    
    # Find the most recent log file
    local latest_log=$(ls -t $log_file 2>/dev/null | head -1)
    
    if [ -f "$latest_log" ]; then
        # Find the last line with COMPLETE epoch results (has val_accuracy AND val_loss)
        local summary_line=$(grep -E "step - accuracy:.*val_accuracy:.*val_loss:" "$latest_log" | tail -1)
        
        if [ -n "$summary_line" ]; then
            # Extract epoch number from the preceding "Epoch X/1000" line
            local line_num=$(grep -n "step - accuracy:.*val_accuracy:.*val_loss:" "$latest_log" | tail -1 | cut -d: -f1)
            local epoch=$(head -n $line_num "$latest_log" | grep -E "^Epoch [0-9]+/" | tail -1 | grep -oP "Epoch \K[0-9]+(?=/)")
            
            # Extract validation accuracy and loss
            local val_acc=$(echo "$summary_line" | grep -oP "val_accuracy: \K[0-9.]+")
            local val_loss=$(echo "$summary_line" | grep -oP "val_loss: \K[0-9.]+")
            
            if [ -n "$val_acc" ] && [ -n "$val_loss" ]; then
                # Convert to percentage and format
                local val_acc_pct=$(echo "$val_acc * 100" | LC_NUMERIC=C bc 2>/dev/null | LC_NUMERIC=C awk '{printf "%.2f", $1}')
                local val_loss_fmt=$(echo "$val_loss" | LC_NUMERIC=C awk '{printf "%.4f", $1}')
                
                echo -e "${GREEN}$model${NC}: Epoch ${YELLOW}$epoch${NC} | Val Acc: ${CYAN}${val_acc_pct}%${NC} | Val Loss: ${CYAN}${val_loss_fmt}${NC}"
            else
                echo -e "${YELLOW}$model${NC}: Epoch ${epoch} (in progress...)"
            fi
        else
            # No complete epoch yet, check if training started
            if grep -q "^Epoch 1/" "$latest_log"; then
                echo -e "${YELLOW}$model${NC}: Epoch 1 (in progress...)"
            else
                echo -e "${YELLOW}$model${NC}: Starting..."
            fi
        fi
    else
        echo -e "${RED}$model${NC}: Not started"
    fi
}

# Function to check if training is active
is_training_active() {
    ps aux | grep -E "(CNN|RNN|CRNN|Attention_CRNN|MultiTask_CNN)_Trainer.py" | grep -v grep > /dev/null
    return $?
}

# Function to get pipeline stage
get_pipeline_stage() {
    if tail -50 "$LOG_FILE" | grep -q "STEP 2/5: Model Training"; then
        if is_training_active; then
            echo "🔄 Training Models"
        else
            echo "✅ Training Complete"
        fi
    elif tail -50 "$LOG_FILE" | grep -q "STEP 3/5: Performance Calculations"; then
        echo "📊 Calculating Performance"
    elif tail -50 "$LOG_FILE" | grep -q "STEP 4/5: Visualizations"; then
        echo "📈 Generating Visualizations"
    elif tail -50 "$LOG_FILE" | grep -q "Calibrating optimal thresholds"; then
        echo "🎯 Calibrating Thresholds"
    elif tail -50 "$LOG_FILE" | grep -q "PIPELINE COMPLETED"; then
        echo "✅ COMPLETED"
    else
        echo "⏳ Initializing"
    fi
}

# Function to estimate time remaining
estimate_time_remaining() {
    local model=$1
    local log_file="logs/${model}_training_*.log"
    
    # Find the most recent log file
    local latest_log=$(ls -t $log_file 2>/dev/null | head -1)
    
    if [ -f "$latest_log" ]; then
        # Extract the last completed epoch
        local last_epoch_line=$(grep -E "^Epoch [0-9]+/[0-9]+" "$latest_log" | tail -1)
        
        if [ -n "$last_epoch_line" ]; then
            local current_epoch=$(echo "$last_epoch_line" | grep -oP "Epoch \K[0-9]+(?=/)")
            local min_epochs=50
            
            if [ $current_epoch -lt $min_epochs ]; then
                local remaining=$((min_epochs - current_epoch))
                local time_per_epoch=30  # seconds (approximate)
                local eta=$((remaining * time_per_epoch / 60))
                echo "${eta}min"
            else
                echo "Finishing..."
            fi
        else
            echo "N/A"
        fi
    else
        echo "N/A"
    fi
}

# Set locale for consistent number formatting
export LC_NUMERIC=C

# Main monitoring loop
clear
while true; do
    # Clear screen properly
    clear
    
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  TRAINING PROGRESS MONITOR${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo -e "Log: ${YELLOW}$(basename $LOG_FILE)${NC}"
    echo -e "Time: ${YELLOW}$(date '+%H:%M:%S')${NC}"
    echo ""
    
    # Current stage
    STAGE=$(get_pipeline_stage)
    echo -e "Stage: ${BLUE}$STAGE${NC}"
    echo ""
    
    # Training progress
    if is_training_active || [ -f "0 - DADS dataset extraction/results/cnn_history.csv" ]; then
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${CYAN}  MODEL TRAINING PROGRESS${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        
        get_training_progress "cnn"
        echo -e "  ETA: ~$(estimate_time_remaining cnn)"
        echo ""
        
        get_training_progress "rnn"
        echo -e "  ETA: ~$(estimate_time_remaining rnn)"
        echo ""
        
        get_training_progress "crnn"
        echo -e "  ETA: ~$(estimate_time_remaining crnn)"
        echo ""
        
        get_training_progress "attention_crnn"
        echo -e "  ETA: ~$(estimate_time_remaining attention_crnn)"
        echo ""
    fi
    
    # Recent activity
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  RECENT ACTIVITY${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    tail -5 "$LOG_FILE" | grep -E "\[INFO\]|\[SUCCESS\]|\[ERROR\]|\[✓\]|Epoch" | tail -3
    echo ""
    
    # Check if pipeline is done
    if tail -10 "$LOG_FILE" | grep -q "PIPELINE COMPLETED"; then
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}  ✅ PIPELINE COMPLETED!${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo "Check results:"
        echo "  • cat '6 - Visualization/outputs/performance_by_distance.csv'"
        echo "  • firefox logs/report_*.html"
        break
    fi
    
    # Show active processes
    if is_training_active; then
        echo -e "${YELLOW}Active Trainers:${NC}"
        ps aux | grep -E "(CNN|RNN|CRNN|Attention_CRNN|MultiTask_CNN)_Trainer.py" | grep -v grep | awk '{print "  • " $11}' | sed 's|.*/||'
        echo ""
    fi
    
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "Press Ctrl+C to exit monitoring"
    echo ""
    
    sleep 5
done
