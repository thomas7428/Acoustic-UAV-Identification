# Quick Start Guide - Training New SOTA Models

## ✅ Verification Complete
All integration checks passed! The 4 new SOTA models are ready to train.

## 🚀 Recommended Training Sequence

### Option 1: Train One Model at a Time (Recommended for first run)

#### 1️⃣ Start with EfficientNet (Best Expected F1)
```bash
./run_full_pipeline.sh --skip-dataset --skip-features --models EFFICIENTNET
```
**Expected time**: ~1.5 hours (50 epochs)
**Expected F1**: ~0.96

#### 2️⃣ Then MobileNet (Fastest Inference)
```bash
./run_full_pipeline.sh --skip-dataset --skip-features --models MOBILENET
```
**Expected time**: ~1 hour (lightweight architecture)
**Expected F1**: ~0.94

#### 3️⃣ Then Conformer (SOTA)
```bash
./run_full_pipeline.sh --skip-dataset --skip-features --models CONFORMER
```
**Expected time**: ~2 hours (more complex)
**Expected F1**: ~0.97 (best expected)

#### 4️⃣ Finally TCN (RNN Alternative)
```bash
./run_full_pipeline.sh --skip-dataset --skip-features --models TCN
```
**Expected time**: ~1 hour
**Expected F1**: ~0.90

---

### Option 2: Train All New Models at Once
```bash
./run_full_pipeline.sh --skip-dataset --skip-features \
    --models EFFICIENTNET,MOBILENET,CONFORMER,TCN
```
**Expected time**: ~6 hours sequential
**Note**: Uses existing features, trains 4 models sequentially

---

### Option 3: Train All 8 Models (Full Pipeline)
```bash
./run_full_pipeline.sh
```
**Expected time**: ~8-10 hours sequential
**Includes**: All 4 original + 4 new models

---

### Option 4: Parallel Training (Faster, More CPU)
```bash
./run_full_pipeline.sh --parallel \
    --models EFFICIENTNET,MOBILENET,CONFORMER,TCN
```
**Expected time**: ~2-3 hours (parallel execution)
**CPU usage**: High (4 models training simultaneously)
**RAM usage**: ~15-20 GB

---

## 📊 Monitor Training

### Check Training Progress
```bash
# View real-time logs
tail -f logs/pipeline_*.log

# Check model files
ls -lh "0 - DADS dataset extraction/saved_models/"

# Check history files
ls -lh "0 - DADS dataset extraction/results/"*history.csv
```

### Expected Output Files (per model)
- `saved_models/efficientnet_model.keras` (~5 MB)
- `saved_models/mobilenet_model.keras` (~2 MB)
- `saved_models/conformer_model.keras` (~8 MB)
- `saved_models/tcn_model.keras` (~3 MB)

- `results/efficientnet_history.csv` (training curves)
- `results/efficientnet_accuracy.json` (test metrics)

---

## 🎯 After Training: Calibration & Testing

### 1. Calibrate Thresholds (Critical!)
```bash
python "3 - Single Model Performance Calculation/calibrate_thresholds.py" --all-models
```

This will calibrate optimal decision thresholds for all 8 models.

### 2. Test Individual Model
```bash
python "3 - Single Model Performance Calculation/Universal_Perf_Tester.py" \
    --model EFFICIENTNET --split test
```

### 3. Compare All Models
```bash
python "6 - Visualization/run_visualizations.py"
```

This generates comparison plots in `6 - Visualization/outputs/`:
- Performance heatmap (all models)
- ROC curves
- Precision-Recall curves
- Confusion matrices

---

## 🔍 Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size in config.py
# Change BATCH_SIZE from 128 to 64 or 32
nano config.py
```

### TensorFlow/CUDA Warnings
These are normal and can be ignored:
```
I tensorflow/core/platform/cpu_feature_guard.cc:...
W tensorflow/compiler/xla/stream_executor/...
```

The models run on CPU and work correctly.

### Model Training Fails
```bash
# Check if features exist
ls -lh "0 - DADS dataset extraction/extracted_features/"

# Re-extract features if needed
cd "1 - Preprocessing and Features Extraction"
python Mel_Preprocess_and_Feature_Extract.py --split train
python Mel_Preprocess_and_Feature_Extract.py --split val
python Mel_Preprocess_and_Feature_Extract.py --split test
```

---

## 📈 Expected Performance Improvements

Current best model: **Attention-CRNN** (F1 = 0.93)

Expected new best models:
1. **Conformer**: F1 ~0.97 (+4.3% improvement) - **SOTA**
2. **EfficientNet**: F1 ~0.96 (+3.2% improvement)
3. **MobileNet**: F1 ~0.94 (+1% improvement, **fastest**)
4. **TCN**: F1 ~0.90 (faster than RNN)

---

## 🎓 Training Tips

1. **Start with one model** to verify everything works
2. **Monitor RAM usage**: `htop` or `watch -n 1 free -h`
3. **Use --parallel** only if you have >16 GB RAM available
4. **Check logs** regularly for errors or warnings
5. **Backup models** after successful training

---

## 🎉 Ready to Start!

**Recommended first command**:
```bash
./run_full_pipeline.sh --skip-dataset --skip-features --models EFFICIENTNET
```

This will:
1. ✅ Skip dataset preparation (use existing)
2. ✅ Skip feature extraction (use existing NPZ files)
3. ✅ Train EfficientNet model (~1.5 hours)
4. ✅ Calibrate threshold for EfficientNet
5. ✅ Test EfficientNet on train/val/test
6. ✅ Generate visualizations

**Let's train the first model and see the results!** 🚀
