#!/bin/bash
################################################################################
# Acoustic UAV Detection - Complete Pipeline (v3.1)
# 
# Pipeline modernisé reflétant la nouvelle architecture:
# 1. Préparation du dataset (master_setup_v2.py)
# 2. Extraction des features MEL (Mel_Preprocess_and_Feature_Extract.py)
# 3. Entraînement des modèles (CNN, RNN, CRNN, ATTENTION_CRNN)
# 4. Calibration des thresholds (calibrate_thresholds.py - multi-critères)
# 5. Évaluation des performances (Universal_Perf_Tester.py)
# 6. Visualisations (run_visualizations.py)
#
# Usage:
#   ./run_full_pipeline.sh [OPTIONS]
#
# Options:
#   --download-offline-dads  Download full DADS dataset for offline use (run once)
#   --skip-dataset        Skip dataset preparation (reuse existing)
#   --skip-features       Skip feature extraction (reuse existing MEL files)
#   --skip-training       Skip model training (reuse existing models)
#   --skip-calibration    Skip threshold calibration (use existing thresholds)
#   --skip-testing        Skip performance testing (reuse existing results)
#   --models MODEL1,MODEL2  Train only specific models (CNN,RNN,CRNN,ATTENTION_CRNN)
#   --parallel            Train models in parallel (faster but more CPU)
#   --skip-viz            Skip visualizations
#   --use-class-aware-calibration  Use legacy class-aware calibration (deprecated)
#   --skip-pre-calib-eval  Skip baseline evaluation (legacy mode)
#   --skip-post-calib-eval Skip post-calibration evaluation (legacy mode)
#   --help                Show this help message
#
# Examples:
#   ./run_full_pipeline.sh --parallel                        # Full pipeline
#   ./run_full_pipeline.sh --skip-dataset --models CNN,RNN   # Train specific models
#   ./run_full_pipeline.sh --skip-training --skip-calibration  # Reuse models/thresholds
#   ./run_full_pipeline.sh --download-offline-dads           # Download DADS once
################################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default settings
SKIP_DATASET=false
SKIP_FEATURES=false
SKIP_TRAINING=false
SKIP_TESTING=false
MODELS="CNN,RNN,CRNN,ATTENTION_CRNN"
PARALLEL=false
THRESHOLDS="0.5"
SKIP_VIZ=false
USE_CLASS_AWARE_CAL=false
SKIP_PRE_CALIB_EVAL=false
SKIP_CALIBRATION=false
SKIP_POST_CALIB_EVAL=false
TRAINER_FLAGS=""
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_DIR/.venv/bin/python"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"
# Determine default number of workers for evaluation (N-1 cores)
CPU_COUNT=$(nproc 2>/dev/null || echo 1)
DEFAULT_WORKERS=$(( CPU_COUNT > 1 ? CPU_COUNT - 1 : 1 ))
# Cap to avoid oversubscription
if [ $DEFAULT_WORKERS -gt 8 ]; then
    DEFAULT_WORKERS=8
fi
WORKERS="$DEFAULT_WORKERS"

# Check for --no-nohup flag first (before parsing other args)
USE_NOHUP=true
for arg in "$@"; do
    if [ "$arg" = "--no-nohup" ]; then
        USE_NOHUP=false
        break
    fi
done

# Preserve the original arguments so we can re-launch with the same flags
ORIGINAL_ARGS=("$@")

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --download-offline-dads)
            echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${BLUE}║      DOWNLOADING FULL DADS DATASET (OFFLINE MODE)         ║${NC}"
            echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            cd "$PROJECT_DIR/0 - DADS dataset extraction"
            bash download_full_dads_offline.sh
            echo ""
            echo -e "${GREEN}✓ Offline dataset downloaded successfully${NC}"
            echo -e "${GREEN}  Future runs will automatically use the offline dataset${NC}"
            echo ""
            exit 0
            ;;
        --skip-dataset)
            SKIP_DATASET=true
            shift
            ;;
        --skip-features)
            SKIP_FEATURES=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-testing)
            SKIP_TESTING=true
            shift
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --thresholds)
            THRESHOLDS="$2"
            shift 2
            ;;
        --skip-viz)
            SKIP_VIZ=true
            shift
            ;;
        --skip-pre-calib-eval)
            SKIP_PRE_CALIB_EVAL=true
            shift
            ;;
        --skip-calibration)
            SKIP_CALIBRATION=true
            shift
            ;;
        --skip-post-calib-eval)
            SKIP_POST_CALIB_EVAL=true
            shift
            ;;
        --trainer-flags)
            TRAINER_FLAGS="$2"
            shift 2
            ;;
        --use-class-aware-calibration)
            USE_CLASS_AWARE_CAL=true
            shift
            ;;
        --no-nohup)
            # Internal flag, already processed
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

# Create log directory
mkdir -p "$LOG_DIR"

# PID file for tracking
PID_FILE="$PROJECT_DIR/run_pipeline.pid"

# Relaunch under nohup if not already
if [ "$USE_NOHUP" = true ] && [ -z "${NOHUP_LAUNCHED:-}" ]; then
    echo "Re-launching under nohup → log: $LOG_FILE"
    # Use the preserved original args so flags like --skip-* are forwarded
    # Use absolute path to this script to ensure nohup can execute it from any cwd
    SCRIPT_ABS="$PROJECT_DIR/$(basename "$0")"
    env NOHUP_LAUNCHED=1 nohup "$SCRIPT_ABS" "${ORIGINAL_ARGS[@]}" --no-nohup > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Started background process PID=$(cat $PID_FILE)"
    exit 0
fi

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    case $level in
        INFO)
            echo -e "${GREEN}[$timestamp][INFO]${NC} $message"
            ;;
        WARN)
            echo -e "${YELLOW}[$timestamp][WARN]${NC} $message"
            ;;
        ERROR)
            echo -e "${RED}[$timestamp][ERROR]${NC} $message"
            ;;
        STEP)
            echo -e "${CYAN}[$timestamp][STEP]${NC} $message"
            ;;
        SUCCESS)
            echo -e "${GREEN}[$timestamp][✓]${NC} $message"
            ;;
        *)
            echo -e "[$timestamp] $message"
            ;;
    esac
}

# Print header
print_header() {
    echo -e "${MAGENTA}"
    echo "================================================================================"
    echo "  🚀 ACOUSTIC UAV DETECTION - FULL PIPELINE V3.0"
    echo "================================================================================"
    echo -e "${NC}"
    log INFO "Timestamp: $TIMESTAMP"
    log INFO "Log File: $LOG_FILE"
    log INFO "Skip Dataset: $SKIP_DATASET"
    log INFO "Skip Features: $SKIP_FEATURES"
    log INFO "Skip Training: $SKIP_TRAINING"
    log INFO "Skip Testing: $SKIP_TESTING"
    log INFO "Models: $MODELS"
    log INFO "Parallel: $PARALLEL"
    log INFO "Thresholds: $THRESHOLDS"
    log INFO "Skip Viz: $SKIP_VIZ"
    log INFO "Trainer Flags: $TRAINER_FLAGS"
    echo ""
}

# Check prerequisites
check_prerequisites() {
    log STEP "Checking prerequisites..."
    
    if [ ! -f "$VENV_PATH" ]; then
        log ERROR "Python venv not found: $VENV_PATH"
        exit 1
    fi
    log SUCCESS "Python venv OK"
    
    # Verify critical scripts exist
    local required_scripts=(
        "0 - DADS dataset extraction/master_setup_v2.py"
        "1 - Preprocessing and Features Extraction/Mel_Preprocess_and_Feature_Extract.py"
        "2 - Model Training/CNN_Trainer.py"
        "2 - Model Training/RNN_Trainer.py"
        "2 - Model Training/CRNN_Trainer.py"
        "2 - Model Training/Attention_CRNN_Trainer.py"
        "3 - Single Model Performance Calculation/Universal_Perf_Tester.py"
        "3 - Single Model Performance Calculation/calibrate_thresholds.py"
        "6 - Visualization/run_visualizations.py"
    )
    
    for script in "${required_scripts[@]}"; do
        if [ ! -f "$PROJECT_DIR/$script" ]; then
            log ERROR "Missing critical script: $script"
            exit 1
        fi
    done
    log SUCCESS "All scripts found"
}

# Step 1: Dataset preparation
prepare_dataset() {
    if [ "$SKIP_DATASET" = true ]; then
        log STEP "Skipping dataset preparation (--skip-dataset)"
        return
    fi
    
    log STEP "STEP 1/5: Dataset Preparation"
    log INFO "Running master_setup_v2.py..."
    
    cd "$PROJECT_DIR/0 - DADS dataset extraction"
    
    if ! "$VENV_PATH" master_setup_v2.py; then
        log ERROR "Dataset preparation failed"
        exit 1
    fi
    
    cd "$PROJECT_DIR"
    log SUCCESS "Dataset prepared successfully"
}

# Step 2: Feature extraction
extract_features() {
    if [ "$SKIP_FEATURES" = true ]; then
        log STEP "Skipping feature extraction (--skip-features)"
        return
    fi
    
    log STEP "STEP 2/5: MEL Feature Extraction"
    log INFO "Extracting features for train/val/test splits..."
    
    cd "$PROJECT_DIR/1 - Preprocessing and Features Extraction"
    
    for split in train val test; do
        log INFO "Extracting MEL features for $split..."
        if ! "$VENV_PATH" Mel_Preprocess_and_Feature_Extract.py --split "$split"; then
            log ERROR "Feature extraction failed for $split"
            exit 1
        fi
    done
    
    cd "$PROJECT_DIR"
    log SUCCESS "All features extracted"
}

# Step 3: Model training
train_models() {
    if [ "$SKIP_TRAINING" = true ]; then
        log STEP "Skipping model training (--skip-training)"
        return
    fi
    
    log STEP "STEP 3/5: Model Training"
    
    # Convert comma-separated list to array
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
    
    if [ "$PARALLEL" = true ]; then
        log INFO "Training models in PARALLEL: ${MODEL_ARRAY[*]}"
        
        local pids=()
        for model in "${MODEL_ARRAY[@]}"; do
            train_single_model "$model" &
            pids+=($!)
        done
        
        # Wait for all to complete
        local failed=0
        for i in "${!pids[@]}"; do
            if ! wait "${pids[$i]}"; then
                log ERROR "Model ${MODEL_ARRAY[$i]} training failed"
                failed=1
            fi
        done
        
        if [ $failed -eq 1 ]; then
            log ERROR "Some models failed to train"
            exit 1
        fi
    else
        log INFO "Training models SEQUENTIALLY: ${MODEL_ARRAY[*]}"
        
        for model in "${MODEL_ARRAY[@]}"; do
            train_single_model "$model"
        done
    fi
    
    log SUCCESS "All models trained"
}

# Train a single model
train_single_model() {
    local model=$1
    local model_lower=$(echo "$model" | tr '[:upper:]' '[:lower:]')
    
    log INFO "Training $model..."
    
    cd "$PROJECT_DIR/2 - Model Training"
    
    case $model in
        CNN)
            trainer="CNN_Trainer.py"
            ;;
        RNN)
            trainer="RNN_Trainer.py"
            ;;
        CRNN)
            trainer="CRNN_Trainer.py"
            ;;
        ATTENTION_CRNN)
            trainer="Attention_CRNN_Trainer.py"
            ;;
        *)
            log ERROR "Unknown model: $model"
            return 1
            ;;
    esac
    
    # Forward any trainer-specific flags (e.g. --min_epochs, --max_epochs, --use_dynamic_weight, --dyn_min_w, --dyn_max_w, --batch_size)
    if ! eval "\"$VENV_PATH\" \"$trainer\" $TRAINER_FLAGS"; then
        log ERROR "$model training failed"
        return 1
    fi
    
    cd "$PROJECT_DIR"
    log SUCCESS "$model trained successfully"
}

# Step 4: Threshold Calibration (after training, before testing)
calibrate_thresholds() {
    if [ "$SKIP_CALIBRATION" = true ]; then
        log STEP "Skipping threshold calibration (--skip-calibration)"
        return
    fi
    
    log STEP "STEP 4/6: Threshold Calibration (Multi-Criteria Hierarchical)"
    
    # Convert comma-separated list to array
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
    
    local calibrator="$PROJECT_DIR/3 - Single Model Performance Calculation/calibrate_thresholds.py"
    
    if [ ! -f "$calibrator" ]; then
        log ERROR "Calibration script not found: $calibrator"
        return 1
    fi
    
    local failed_calibrations=0
    
    log INFO "Running multi-criteria threshold calibration..."
    log INFO "Criteria: Tier 1 (Constraints) → Tier 2 (Balanced Precision) → Tier 3 (F1/Recall)"
    echo ""
    
    # Build model list for calibrator
    CALIB_MODELS=$(echo "$MODELS" | tr ',' ' ')
    
    for model in $CALIB_MODELS; do
        log INFO "  • Calibrating $model thresholds..."
        if "$VENV_PATH" "$calibrator" --model "$model" 2>&1 | grep -v "cuda\|CUDA\|GPU"; then
            log SUCCESS "  ✓ $model calibration complete"
        else
            log WARN "  ✗ $model calibration failed"
            ((failed_calibrations++))
        fi
        echo ""
    done
    
    if [ $failed_calibrations -eq 0 ]; then
        log SUCCESS "All threshold calibrations completed successfully"
    else
        log WARN "Threshold calibration completed with $failed_calibrations issue(s)"
    fi
    
    # Show calibrated thresholds summary
    local calib_file="$PROJECT_DIR/0 - DADS dataset extraction/results/calibrated_thresholds.json"
    if [ -f "$calib_file" ]; then
        log INFO "Calibrated thresholds saved to: $calib_file"
    fi
}

# Step 5: Performance calculation with calibrated thresholds
calculate_performance() {
    if [ "$SKIP_TESTING" = true ]; then
        log STEP "Skipping performance testing (--skip-testing)"
        return
    fi
    
    log STEP "STEP 5/6: Performance Evaluation (Using Calibrated Thresholds)"
    
    # Convert comma-separated list to array
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
    
    local calibrator="$PROJECT_DIR/3 - Single Model Performance Calculation/calibrate_thresholds.py"
    local tester="$PROJECT_DIR/3 - Single Model Performance Calculation/Universal_Perf_Tester.py"
    
    if [ ! -f "$calibrator" ] || [ ! -f "$tester" ]; then
        log ERROR "Required scripts not found"
        return 1
    fi
    
    local failed_tests=0

    # LEGACY CLASS-AWARE WORKFLOW (deprecated but kept for compatibility)
    if [ "$USE_CLASS_AWARE_CAL" = true ]; then
        log STEP "Using class-aware calibration workflow"

        # 1) Baseline run: generate predictions/scores with default threshold 0.5
        CALIB_FILE="$PROJECT_DIR/0 - DADS dataset extraction/results/calibrated_thresholds.json"
        if [ "$SKIP_PRE_CALIB_EVAL" = true ]; then
            log STEP "Skipping pre-calibration baseline evaluation (--skip-pre-calib-eval)"
        else
            log INFO "[1/3] Baseline test run (threshold=0.5) for all models/splits"
            # Remove any existing calibrated thresholds so Universal_Perf_Tester does not pick them up
            if [ -f "$CALIB_FILE" ]; then
                log INFO "Removing existing calibration file before baseline run: $CALIB_FILE"
                rm -f "$CALIB_FILE"
            fi
            for model in "${MODEL_ARRAY[@]}"; do
                for split in val test; do
                    log INFO "  • Baseline: $model on $split"
                    if ! "$VENV_PATH" "$tester" --model "$model" --split "$split" --threshold 0.5 --workers "$WORKERS" 2>&1 | grep -v "cuda\|CUDA\|GPU"; then
                        log WARN "    ✗ Baseline test failed: $model $split"
                        ((failed_tests++))
                    fi
                done
            done
        fi

        # 2) Calibrate thresholds using class-aware calibration (saves visualizer JSON)
        if [ "$SKIP_CALIBRATION" = true ]; then
            log STEP "Skipping calibration step (--skip-calibration)"
            if [ ! -f "$CALIB_FILE" ]; then
                log ERROR "Calibration file not found: $CALIB_FILE (cannot skip calibration without existing file)"
                exit 1
            fi
            log INFO "Using existing calibration file: $CALIB_FILE"
        else
            log INFO "[2/3] Running class-aware calibration (relaxed defaults)"
            if ! "$VENV_PATH" "$PROJECT_DIR/calibrate_thresholds.py" --mode class_aware --min-prec-drone 0.7 --min-prec-ambient 0.85 --class-alpha 0.4 --max-class-acc-gap 0.2 --save 2>&1 | grep -v "cuda\|CUDA\|GPU"; then
                log WARN "Calibration script reported warnings/errors"
            else
                log SUCCESS "Calibration completed and thresholds saved"
            fi
        fi

        # 3) Re-run tests using calibrated thresholds (Universal_Perf_Tester defaults to config.MODEL_THRESHOLDS)
        if [ "$SKIP_POST_CALIB_EVAL" = true ]; then
            log STEP "Skipping post-calibration evaluation (--skip-post-calib-eval)"
        else
            log INFO "[3/3] Re-running tests with calibrated thresholds from config"
            for model in "${MODEL_ARRAY[@]}"; do
                for split in train val test; do
                    log INFO "  • Calibrated test: $model on $split"
                    if ! "$VENV_PATH" "$tester" --model "$model" --split "$split" --workers "$WORKERS" 2>&1 | grep -v "cuda\|CUDA\|GPU"; then
                        log WARN "    ✗ Calibrated test failed: $model $split"
                        ((failed_tests++))
                    fi
                done
            done
        fi

        if [ $failed_tests -eq 0 ]; then
            log SUCCESS "Class-aware calibration workflow completed successfully"
        else
            log WARN "Class-aware calibration workflow completed with $failed_tests issues"
        fi
        return
    fi

    # Test with calibrated thresholds on all splits
    log INFO "Testing models with calibrated thresholds on all splits..."
    for model in "${MODEL_ARRAY[@]}"; do
        for split in train val test; do
            log INFO "  • Testing $model on $split..."
            
            if "$VENV_PATH" "$tester" \
                --model "$model" \
                --split "$split" \
                --workers "$WORKERS" 2>&1 | \
                grep -E "(Dataset|Test en cours|Loaded calibrated|Accuracy|Résultats sauvegardés)" | \
                grep -v "cuda\|CUDA\|GPU" > /dev/null; then
                log SUCCESS "  ✓ $model $split"
            else
                log WARN "  ✗ Test failed: $model $split"
                ((failed_tests++))
            fi
        done
        echo ""
    done

    if [ $failed_tests -eq 0 ]; then
        log SUCCESS "All performance tests completed successfully"
    else
        log WARN "Performance calculation completed with $failed_tests issues"
    fi
}

# Step 6: Visualizations (Enhanced Pipeline)
generate_visualizations() {
    if [ "$SKIP_VIZ" = true ]; then
        log STEP "Skipping visualizations (--skip-viz)"
        return
    fi
    
    log STEP "STEP 6/6: Generating Visualizations"
    
    local viz_script="$PROJECT_DIR/6 - Visualization/run_visualizations.py"
    
    if [ ! -f "$viz_script" ]; then
        log ERROR "Visualization script not found: $viz_script"
        return 1
    fi
    
    log INFO "Running unified visualization pipeline..."
    
    if ! "$VENV_PATH" "$viz_script" 2>&1 | grep -v "cuda\|CUDA\|GPU"; then
        log WARN "Visualization generation had warnings (non-fatal)"
    else
        log SUCCESS "Enhanced visualizations generated successfully"
    fi
}

# Generate final report
generate_report() {
    log STEP "Generating Pipeline Report..."
    
    local report_file="$LOG_DIR/pipeline_report_${TIMESTAMP}.txt"
    
    {
        echo "================================================================================"
        echo "  ACOUSTIC UAV DETECTION - PIPELINE EXECUTION REPORT"
        echo "================================================================================"
        echo ""
        echo "Execution Details:"
        echo "  Timestamp: $TIMESTAMP"
        echo "  Duration: $(date -ud "@$(($(date +%s) - $(date -d "$(echo $TIMESTAMP | sed 's/_/ /;s/\(..\)\(..\)\(..\)$/\1:\2:\3/')" +%s)))" +%H:%M:%S)"
        echo "  Log File: $LOG_FILE"
        echo ""
        echo "Configuration:"
        echo "  Dataset Prepared: $([ "$SKIP_DATASET" = false ] && echo "Yes" || echo "No (skipped)")"
        echo "  Features Extracted: $([ "$SKIP_FEATURES" = false ] && echo "Yes" || echo "No (skipped)")"
        echo "  Models Trained: $MODELS"
        echo "  Training Mode: $([ "$PARALLEL" = true ] && echo "Parallel" || echo "Sequential")"
        echo "  Thresholds Tested: $THRESHOLDS"
        echo "  Visualizations: $([ "$SKIP_VIZ" = false ] && echo "Generated" || echo "Skipped")"
        echo ""
        echo "Output Locations:"
        echo "  Models: 0 - DADS dataset extraction/saved_models/"
        echo "  Performance: 0 - DADS dataset extraction/results/performance/"
        echo "  Visualizations: 6 - Visualization/outputs/"
        echo "  Training History: 0 - DADS dataset extraction/results/*_history.csv"
        echo ""
        echo "Next Steps:"
        echo "  1. Review performance results: 6 - Visualization/outputs/performance_summary.txt"
        echo "  2. Check visualizations: 6 - Visualization/outputs/"
        echo "  3. Analyze threshold impact: 6 - Visualization/outputs/threshold_calibration_*.png"
        echo "  4. Use quick_viz.py for custom visualizations"
        echo ""
    } > "$report_file"
    
    cat "$report_file"
    log SUCCESS "Report saved: $report_file"
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    print_header
    check_prerequisites
    
    prepare_dataset
    extract_features
    train_models
    calibrate_thresholds
    calculate_performance
    generate_visualizations
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo ""
    echo -e "${MAGENTA}"
    echo "================================================================================"
    echo "  ✓ PIPELINE COMPLETED SUCCESSFULLY"
    echo "================================================================================"
    echo -e "${NC}"
    log SUCCESS "Total duration: $(date -ud "@$duration" +%H:%M:%S)"
    
    generate_report
    
    # Cleanup PID file
    rm -f "$PID_FILE"
}

# Run main
main "$@"
