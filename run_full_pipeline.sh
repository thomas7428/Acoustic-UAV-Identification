#!/bin/bash
################################################################################
# Acoustic UAV Detection - Full Pipeline Automation Script
# 
# This script orchestrates the complete ML pipeline:
# 1. Dataset generation (optional)
# 2. Parallel model training (CNN + RNN + CRNN)
# 3. Performance calculations
# 4. Visualizations generation
# 5. Report generation
#
# Usage:
#   ./run_full_pipeline.sh [OPTIONS]
#
# Options:
#   --config PATH         Path to augmentation config (default: augment_config_v2.json)
#   --skip-dataset        Skip dataset generation (reuse existing, NO cleanup)
#   --parallel            Train models in parallel (faster)
#   --no-visualizations   Skip visualization generation
#   --keep-logs           Keep existing logs (default: delete old logs)
#   --help                Show this help message
#
# Important:
#   By DEFAULT, the pipeline will:
#   - CLEANUP existing datasets before generating new ones
#   - DELETE old training logs and history CSVs
#   Use --skip-dataset to keep existing data and --keep-logs to preserve logs.
#
# Examples:
#   ./run_full_pipeline.sh --parallel                    # Full pipeline with cleanup
#   ./run_full_pipeline.sh --skip-dataset --parallel     # Reuse existing (no cleanup)
#   ./run_full_pipeline.sh --keep-logs --parallel        # Keep old logs
#   ./run_full_pipeline.sh --config custom_config.json   # Custom config with cleanup
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default settings
CONFIG_FILE="0 - DADS dataset extraction/augment_config_v3.json"
SKIP_DATASET=false
SKIP_FEATURES=false
PARALLEL=false
SKIP_VISUALIZATIONS=false
KEEP_LOGS=false
FORCE_FEATURES=false
FORCE_RETEST=false
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_DIR/.venv/bin/python"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --skip-dataset)
            SKIP_DATASET=true
            shift
            ;;
        --skip-features)
            SKIP_FEATURES=true
            shift
            ;;
        --force-features)
            FORCE_FEATURES=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --no-visualizations)
            SKIP_VISUALIZATIONS=true
            shift
            ;;
        --force-retest)
            FORCE_RETEST=true
            shift
            ;;
        --keep-logs)
            KEEP_LOGS=true
            shift
            ;;
        --help)
            grep "^#" "$0" | grep -v "#!/bin/bash" | sed 's/^# *//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Create log directory (and relaunch under nohup so the process survives SSH disconnects)
mkdir -p "$LOG_DIR"

# PID file for background runs
PID_FILE="$PROJECT_DIR/run_pipeline.pid"

# If not already launched under nohup, relaunch this script with nohup and exit the parent.
# This ensures the process keeps running if the SSH session is closed.
if [ -z "${NOHUP_LAUNCHED:-}" ]; then
    echo "Re-launching script under nohup; log -> $LOG_FILE"
    # Export a marker for the child and run under nohup, redirecting stdout/stderr to the chosen log file
    NOHUP_LAUNCHED=1 nohup "$0" "$@" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Started background process with PID $(cat $PID_FILE) (log: $LOG_FILE)"
    exit 0
fi

# Logging function (must be defined BEFORE first use)
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    case $level in
        INFO)
            echo -e "${GREEN}[INFO]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        STEP)
            echo -e "${CYAN}[STEP]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        SUCCESS)
            echo -e "${GREEN}[✓]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        *)
            echo -e "$message" | tee -a "$LOG_FILE"
            ;;
    esac
}

# Cleanup old logs by default (unless --keep-logs specified)
if [ "$KEEP_LOGS" = false ]; then
    log INFO "Cleaning up old training logs and history files..."
    
    # Remove old training logs
    rm -f "$LOG_DIR"/*_training_*.log 2>/dev/null || true
    
    # Remove old history CSVs from results directory
    rm -f "$PROJECT_DIR/0 - DADS dataset extraction/results/"*_history.csv 2>/dev/null || true
    
    log SUCCESS "Old logs cleaned up"
else
    log INFO "Keeping existing logs (--keep-logs flag)"
fi

# Print header
print_header() {
    echo -e "${MAGENTA}" | tee -a "$LOG_FILE"
    echo "================================================================================" | tee -a "$LOG_FILE"
    echo "  🚀 ACOUSTIC UAV DETECTION - FULL PIPELINE" | tee -a "$LOG_FILE"
    echo "================================================================================" | tee -a "$LOG_FILE"
    echo -e "${NC}" | tee -a "$LOG_FILE"
    log INFO "Timestamp: $TIMESTAMP"
    log INFO "Project Directory: $PROJECT_DIR"
    log INFO "Log File: $LOG_FILE"
    log INFO "Config File: $CONFIG_FILE"
    log INFO "Skip Dataset: $SKIP_DATASET"
    log INFO "Parallel Training: $PARALLEL"
    log INFO "Keep Logs: $KEEP_LOGS"
    log INFO ""
}

# Check prerequisites
check_prerequisites() {
    log STEP "Checking prerequisites..."
    
    # Check Python venv
    if [ ! -f "$VENV_PATH" ]; then
        log ERROR "Python virtual environment not found at $VENV_PATH"
        exit 1
    fi
    log SUCCESS "Python venv found"
    
    # Check config file
    if [ ! -f "$PROJECT_DIR/$CONFIG_FILE" ]; then
        log ERROR "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    log SUCCESS "Config file found"
    
    # Check required directories
    for dir in "0 - DADS dataset extraction" "1 - Preprocessing and Features Extraction" "2 - Model Training" "3 - Single Model Performance Calculation" "6 - Visualization"; do
        if [ ! -d "$PROJECT_DIR/$dir" ]; then
            log ERROR "Required directory not found: $dir"
            exit 1
        fi
    done
    log SUCCESS "All directories present"
    
    log INFO ""
}

# Step 1: Dataset Generation
run_dataset_generation() {
    if [ "$SKIP_DATASET" = true ]; then
        log STEP "Skipping dataset generation (--skip-dataset flag)"
        log WARNING "Using existing datasets - no cleanup performed"
        log INFO ""
        return
    fi
    
    log STEP "STEP 1/5: Dataset Generation & Augmentation"
    log INFO "Running master_setup_v2.py (with cleanup)..."
    log INFO "This will DELETE existing datasets to prevent contamination"
    log INFO "Config File: $CONFIG_FILE"
    
    cd "$PROJECT_DIR/0 - DADS dataset extraction"
    
    # Extract just the filename from CONFIG_FILE path
    CONFIG_BASENAME=$(basename "$CONFIG_FILE")
    
    # Run with cleanup by default (no --no-cleanup flag) and pass config
    if "$VENV_PATH" master_setup_v2.py --config "$CONFIG_BASENAME" 2>&1 | tee -a "$LOG_FILE"; then
        log SUCCESS "Dataset generation completed"
    else
        log ERROR "Dataset generation failed"
        exit 1
    fi
    
    cd "$PROJECT_DIR"
    log INFO ""
}

# Step 1.5: Feature Extraction
run_feature_extraction() {
    log STEP "STEP 1.5/5: Feature Extraction"
    
    cd "$PROJECT_DIR/1 - Preprocessing and Features Extraction"
    
    log INFO "Extracting MEL features..."
    if "$VENV_PATH" Mel_Preprocess_and_Feature_Extract.py 2>&1 | tee -a "$LOG_FILE"; then
        log SUCCESS "MEL feature extraction completed"
    else
        log ERROR "MEL feature extraction failed"
        exit 1
    fi
    
    log INFO "Extracting MFCC features..."
    if "$VENV_PATH" MFCC_Preprocess_and_Feature_Extract.py 2>&1 | tee -a "$LOG_FILE"; then
        log SUCCESS "MFCC feature extraction completed"
    else
        log ERROR "MFCC feature extraction failed"
        exit 1
    fi

    log INFO "Extracting MEL test index from WAVs..."
    if "$VENV_PATH" regenerate_mel_test_index_from_wavs.py 2>&1 | tee -a "$LOG_FILE"; then
        log SUCCESS "MEL test index extraction completed"
    else
        log ERROR "MEL test index extraction failed"
        exit 1
    fi
    
    cd "$PROJECT_DIR"
    log INFO ""
}

# Step 2: Model Training
run_model_training() {
    log STEP "STEP 2/5: Model Training"
    
    cd "$PROJECT_DIR/2 - Model Training"
    
    if [ "$PARALLEL" = true ]; then
        log INFO "Training models in PARALLEL mode (faster)..."
        log INFO "Using staggered start to manage RAM usage..."
        
        # Start trainers with delays to avoid loading 6GB+ features simultaneously
        log INFO "Starting CNN training (min_epochs=50, stratified validation)..."
        "$VENV_PATH" CNN_Trainer.py --min_epochs 50 --stratified-validation > "$LOG_DIR/cnn_training_${TIMESTAMP}.log" 2>&1 &
        CNN_PID=$!
        
        log INFO "Waiting 30s for CNN to load features into RAM..."
        sleep 30
        
        log INFO "Starting RNN training (min_epochs=50, stratified validation)..."
        "$VENV_PATH" RNN_Trainer.py --min_epochs 50 --stratified-validation > "$LOG_DIR/rnn_training_${TIMESTAMP}.log" 2>&1 &
        RNN_PID=$!
        
        log INFO "Waiting 30s for RNN to load features into RAM..."
        sleep 30
        
        log INFO "Starting CRNN training (min_epochs=50, stratified validation)..."
        "$VENV_PATH" CRNN_Trainer.py --min_epochs 50 --stratified-validation > "$LOG_DIR/crnn_training_${TIMESTAMP}.log" 2>&1 &
        CRNN_PID=$!
        
        log INFO "Waiting 30s for CRNN to load features into RAM..."
        sleep 30
        
        log INFO "Starting Attention-Enhanced CRNN training (min_epochs=50, stratified validation)..."
        "$VENV_PATH" Attention_CRNN_Trainer.py --min_epochs 50 --stratified-validation > "$LOG_DIR/attention_crnn_training_${TIMESTAMP}.log" 2>&1 &
        ATTENTION_PID=$!
        
        # Wait for all to complete
        log INFO "Waiting for all models to complete training..."
        
        wait $CNN_PID
        CNN_STATUS=$?
        if [ $CNN_STATUS -eq 0 ]; then
            log SUCCESS "CNN training completed"
        else
            log ERROR "CNN training failed (exit code: $CNN_STATUS)"
        fi
        
        wait $RNN_PID
        RNN_STATUS=$?
        if [ $RNN_STATUS -eq 0 ]; then
            log SUCCESS "RNN training completed"
        else
            log ERROR "RNN training failed (exit code: $RNN_STATUS)"
        fi
        
        wait $CRNN_PID
        CRNN_STATUS=$?
        if [ $CRNN_STATUS -eq 0 ]; then
            log SUCCESS "CRNN training completed"
        else
            log ERROR "CRNN training failed (exit code: $CRNN_STATUS)"
        fi
        
        wait $ATTENTION_PID
        ATTENTION_STATUS=$?
        if [ $ATTENTION_STATUS -eq 0 ]; then
            log SUCCESS "Attention-Enhanced CRNN training completed"
        else
            log ERROR "Attention-Enhanced CRNN training failed (exit code: $ATTENTION_STATUS)"
        fi
        
        # Check if any failed
        if [ $CNN_STATUS -ne 0 ] || [ $RNN_STATUS -ne 0 ] || [ $CRNN_STATUS -ne 0 ] || [ $ATTENTION_STATUS -ne 0 ]; then
            log ERROR "One or more models failed to train"
            exit 1
        fi
        
    else
        log INFO "Training models in SEQUENTIAL mode..."
        
        log INFO "Training CNN (min_epochs=50, stratified validation)..."
        if "$VENV_PATH" CNN_Trainer.py --min_epochs 50 --stratified-validation 2>&1 | tee -a "$LOG_FILE"; then
            log SUCCESS "CNN training completed"
        else
            log ERROR "CNN training failed"
            exit 1
        fi
        
        log INFO "Training RNN (min_epochs=50, stratified validation)..."
        if "$VENV_PATH" RNN_Trainer.py --min_epochs 50 --stratified-validation 2>&1 | tee -a "$LOG_FILE"; then
            log SUCCESS "RNN training completed"
        else
            log ERROR "RNN training failed"
            exit 1
        fi
        
        log INFO "Training CRNN (min_epochs=50, stratified validation)..."
        if "$VENV_PATH" CRNN_Trainer.py --min_epochs 50 --stratified-validation 2>&1 | tee -a "$LOG_FILE"; then
            log SUCCESS "CRNN training completed"
        else
            log ERROR "CRNN training failed"
            exit 1
        fi
        
        log INFO "Training Attention-Enhanced CRNN (min_epochs=50, stratified validation)..."
        if "$VENV_PATH" Attention_CRNN_Trainer.py --min_epochs 50 --stratified-validation 2>&1 | tee -a "$LOG_FILE"; then
            log SUCCESS "Attention-Enhanced CRNN training completed"
        else
            log ERROR "Attention-Enhanced CRNN training failed"
            exit 1
        fi
    fi
    
    cd "$PROJECT_DIR"
    log INFO ""
}

# Step 3: Performance Calculations
run_performance_calculations() {
    log STEP "STEP 3/5: Performance Calculations"
    
    cd "$PROJECT_DIR/3 - Single Model Performance Calculation"
    
    log INFO "Calculating CNN performance..."
    if "$VENV_PATH" CNN_Performance_Calcs.py 2>&1 | tee -a "$LOG_FILE"; then
        log SUCCESS "CNN performance calculated"
    else
        log WARN "CNN performance calculation had warnings"
    fi
    
    log INFO "Calculating RNN performance..."
    if "$VENV_PATH" RNN_Performance_Calcs.py 2>&1 | tee -a "$LOG_FILE"; then
        log SUCCESS "RNN performance calculated"
    else
        log WARN "RNN performance calculation had warnings"
    fi
    
    log INFO "Calculating CRNN performance..."
    if "$VENV_PATH" CRNN_Performance_Calcs.py 2>&1 | tee -a "$LOG_FILE"; then
        log SUCCESS "CRNN performance calculated"
    else
        log WARN "CRNN performance calculation had warnings"
    fi
    
    log INFO "Calculating Attention-Enhanced CRNN performance..."
    if "$VENV_PATH" Attention_CRNN_Performance_Calcs.py 2>&1 | tee -a "$LOG_FILE"; then
        log SUCCESS "Attention-Enhanced CRNN performance calculated"
    else
        log WARN "Attention-Enhanced CRNN performance calculation had warnings"
    fi
    
    log INFO "Converting results for visualization..."
    if "$VENV_PATH" convert_results_for_viz.py 2>&1 | tee -a "$LOG_FILE"; then
        log SUCCESS "Results converted for visualization"
    else
        log WARN "Result conversion had warnings"
    fi
}

# Step 4: Threshold Calibration (NEW - Always run unless skipped)
run_threshold_calibration() {
    log STEP "STEP 4/6: Threshold Calibration"
    
    cd "$PROJECT_DIR/6 - Visualization"
    
    log INFO "Calibrating optimal classification thresholds..."
    if "$VENV_PATH" calibrate_thresholds.py --target-recall 0.85 --save 2>&1 | tee -a "$LOG_FILE"; then
        log SUCCESS "Threshold calibration completed"
        
        # Display results if CSV exists
        if [ -f "outputs/threshold_calibration_report.csv" ]; then
            log INFO "Optimal thresholds:"
            cat outputs/threshold_calibration_report.csv | column -t -s, | tee -a "$LOG_FILE"
        fi
    else
        log WARN "Threshold calibration had issues (continuing pipeline)"
    fi
    
    cd "$PROJECT_DIR"
    log INFO ""
}

# Step 5: Visualizations
run_visualizations() {
    if [ "$SKIP_VISUALIZATIONS" = true ]; then
        log STEP "Skipping visualizations (--no-visualizations flag)"
        log INFO ""
        return
    fi
    
    log STEP "STEP 5/6: Generating Visualizations"
    
    cd "$PROJECT_DIR/6 - Visualization"
    
    log INFO "Running all visualization scripts..."
    VIS_CMD="$VENV_PATH run_all_visualizations.py"
    if [ "$FORCE_RETEST" = true ]; then
        VIS_CMD="$VIS_CMD --force-retest"
    fi

    if eval "$VIS_CMD" 2>&1 | tee -a "$LOG_FILE"; then
        log SUCCESS "All visualizations generated"
    else
        log WARN "Some visualizations may have failed"
    fi
    
    cd "$PROJECT_DIR"
    log INFO ""
}

# Step 6: Generate Report
generate_report() {
    log STEP "STEP 6/6: Generating HTML Report"
    
    REPORT_FILE="$LOG_DIR/report_${TIMESTAMP}.html"
    
    cat > "$REPORT_FILE" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Pipeline Report - $TIMESTAMP</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .success { color: #27ae60; font-weight: bold; }
        .error { color: #e74c3c; font-weight: bold; }
        .info { color: #3498db; }
        .log-section { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; font-family: monospace; font-size: 12px; overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #3498db; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; padding: 15px; background: #ecf0f1; border-radius: 5px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .metric-label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Acoustic UAV Detection Pipeline Report</h1>
        
        <div class="metric">
            <div class="metric-label">Timestamp</div>
            <div class="metric-value">$TIMESTAMP</div>
        </div>
        
        <div class="metric">
            <div class="metric-label">Config</div>
            <div class="metric-value">$(basename "$CONFIG_FILE")</div>
        </div>
        
        <div class="metric">
            <div class="metric-label">Mode</div>
            <div class="metric-value">$([ "$PARALLEL" = true ] && echo "PARALLEL" || echo "SEQUENTIAL")</div>
        </div>
        
        <h2>📊 Pipeline Steps</h2>
        <table>
            <tr><th>Step</th><th>Description</th><th>Status</th></tr>
            <tr><td>1</td><td>Dataset Generation</td><td class="$([ "$SKIP_DATASET" = true ] && echo "info" || echo "success")">$([ "$SKIP_DATASET" = true ] && echo "SKIPPED" || echo "✓ COMPLETED")</td></tr>
            <tr><td>2</td><td>Model Training (CNN, RNN, CRNN, Attention-CRNN)</td><td class="success">✓ COMPLETED</td></tr>
            <tr><td>3</td><td>Performance Calculations</td><td class="success">✓ COMPLETED</td></tr>
            <tr><td>4</td><td>Threshold Calibration</td><td class="success">✓ COMPLETED</td></tr>
            <tr><td>5</td><td>Visualizations</td><td class="$([ "$SKIP_VISUALIZATIONS" = true ] && echo "info" || echo "success")">$([ "$SKIP_VISUALIZATIONS" = true ] && echo "SKIPPED" || echo "✓ COMPLETED")</td></tr>
            <tr><td>6</td><td>Report Generation</td><td class="success">✓ COMPLETED</td></tr>
        </table>
        
        <h2>📁 Output Locations</h2>
        <ul>
            <li><strong>Models:</strong> <code>0 - DADS dataset extraction/saved_models/</code></li>
            <li><strong>Results:</strong> <code>0 - DADS dataset extraction/results/</code></li>
            <li><strong>Visualizations:</strong> <code>6 - Visualization/outputs/</code></li>
            <li><strong>Logs:</strong> <code>logs/</code></li>
        </ul>
        
        <h2>📝 Full Log</h2>
        <div class="log-section">
            <pre>$(cat "$LOG_FILE" | sed 's/\x1b\[[0-9;]*m//g')</pre>
        </div>
        
        <h2>🔗 Quick Links</h2>
        <ul>
            <li><a href="../6 - Visualization/outputs/real_performance_by_distance.png" target="_blank">Performance by Distance</a></li>
            <li><a href="../6 - Visualization/outputs/difficulty_spectrum.png" target="_blank">Difficulty Spectrum</a></li>
            <li><a href="../6 - Visualization/outputs/performance_by_distance.csv" target="_blank">Performance CSV</a></li>
        </ul>
        
        <p style="margin-top: 40px; text-align: center; color: #7f8c8d;">
            Generated by run_full_pipeline.sh on $(date)
        </p>
    </div>
</body>
</html>
EOF
    
    log SUCCESS "HTML report generated: $REPORT_FILE"
    log INFO ""
}

# Main execution
main() {
    print_header
    check_prerequisites
    
    START_TIME=$(date +%s)
    
    if [ "$SKIP_DATASET" = false ]; then
        run_dataset_generation
    fi
    
    # Feature extraction: skip if precomputed features exist unless forced
    if [ "$SKIP_FEATURES" = false ]; then
        FEATURES_DIR="$PROJECT_DIR/0 - DADS dataset extraction/extracted_features"
        if [ "$FORCE_FEATURES" = false ] && [ -d "$FEATURES_DIR" ] && ls "$FEATURES_DIR"/*.json >/dev/null 2>&1; then
            log INFO "Precomputed features detected in $FEATURES_DIR - skipping feature extraction (use --force-features to re-run)"
        else
            run_feature_extraction
        fi
    else
        log INFO "Skipping feature extraction (--skip-features flag)"
    fi
    
    run_model_training
    run_performance_calculations
    run_threshold_calibration
    run_visualizations
    generate_report
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_MIN=$((DURATION / 60))
    DURATION_SEC=$((DURATION % 60))
    
    echo -e "${GREEN}" | tee -a "$LOG_FILE"
    echo "================================================================================" | tee -a "$LOG_FILE"
    echo "  ✓ PIPELINE COMPLETED SUCCESSFULLY!" | tee -a "$LOG_FILE"
    echo "================================================================================" | tee -a "$LOG_FILE"
    echo -e "${NC}" | tee -a "$LOG_FILE"
    log INFO "Total Duration: ${DURATION_MIN}m ${DURATION_SEC}s"
    log INFO "Log File: $LOG_FILE"
    log INFO "Report: $LOG_DIR/report_${TIMESTAMP}.html"
    log INFO ""
    log INFO "Next steps:"
    log INFO "  1. Review report: firefox $LOG_DIR/report_${TIMESTAMP}.html"
    log INFO "  2. Check visualizations: ls -lh '6 - Visualization/outputs/'"
    log INFO "  3. Analyze performance: cat '6 - Visualization/outputs/performance_by_distance.csv'"
}

# Run main function
main
