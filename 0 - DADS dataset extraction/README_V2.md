# Enhanced Dataset Pipeline v2.0

This directory contains enhanced scripts for creating balanced, professional datasets for UAV audio detection with **zero data leakage**.

## 🎯 Key Improvements

### v2.0 Features
1. **Balanced Augmentation**: Both drone AND no-drone classes get augmentation
2. **Train/Test Split**: Proper separation prevents overfitting
3. **Audio Effects**: Pitch shift, time stretch for better generalization
4. **Reproducible**: Fixed random seeds for consistent results
5. **Professional**: Metadata tracking, verification, comprehensive logging

## 📁 File Structure

```
0 - DADS dataset extraction/
├── download_and_prepare_dads.py     # Download DADS dataset from HuggingFace
├── augment_dataset_v2.py            # ✨ NEW: Augment both classes
├── augment_config_v2.json           # ✨ NEW: Enhanced configuration
├── split_dataset.py                 # ✨ NEW: Train/val/test splitter
├── master_setup_v2.py               # ✨ NEW: Automated pipeline
│
├── dataset_test/                    # Original DADS samples
│   ├── 0/                           # No-drone originals
│   └── 1/                           # Drone originals
│
├── dataset_augmented/               # Augmented samples (both classes)
│   ├── 0/                           # Augmented no-drones
│   └── 1/                           # Augmented drones
│
├── dataset_train/                   # Training set (70%)
│   ├── 0/
│   └── 1/
│
├── dataset_val/                     # Validation set (15%)
│   ├── 0/
│   └── 1/
│
└── dataset_test_final/              # Test set (15%) - NEVER seen during training!
    ├── 0/
    └── 1/
```

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Run the complete pipeline
python master_setup_v2.py --drone-samples 100 --no-drone-samples 100

# This will:
# 1. Download DADS dataset
# 2. Augment both classes
# 3. Combine originals + augmented
# 4. Split into train/val/test
# 5. Extract features
```

### Option 2: Manual Step-by-Step

```bash
# Step 1: Download dataset
python download_and_prepare_dads.py --output dataset_test --max-per-class 100

# Step 2: Augment both classes
python augment_dataset_v2.py --config augment_config_v2.json

# Step 3: Combine originals + augmented
python combine_datasets.py --original dataset_test --augmented dataset_augmented --output dataset_combined

# Step 4: Split into train/val/test
python split_dataset.py --source dataset_combined --train 0.7 --val 0.15 --test 0.15
```

## 📊 Configuration

### `augment_config_v2.json`

```json
{
  "drone_augmentation": {
    "enabled": true,
    "categories": [
      "drone_very_far",  // -15dB SNR
      "drone_far",       // -10dB SNR
      "drone_medium",    // -5dB SNR
      "drone_close",     // 0dB SNR
      "drone_very_close" // +5dB SNR
    ]
  },
  
  "no_drone_augmentation": {
    "enabled": true,
    "categories": [
      "ambient_complex",   // 3 sources mixed
      "ambient_moderate",  // 2 sources mixed
      "ambient_simple",    // 1 source with effects
      "ambient_quiet"      // Low volume
    ]
  },
  
  "output": {
    "samples_per_category_drone": 200,
    "samples_per_category_no_drone": 200
  }
}
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `samples_per_category_drone` | 200 | Samples per SNR category for drones |
| `samples_per_category_no_drone` | 200 | Samples per complexity category for no-drones |
| `train_ratio` | 0.7 | 70% for training |
| `val_ratio` | 0.15 | 15% for validation |
| `test_ratio` | 0.15 | 15% for testing |
| `random_seed` | 42 | For reproducibility |

## 🎵 Augmentation Details

### Drone Augmentation (Class 1)
- **Method**: Mix drone audio with background noise at controlled SNR
- **SNR Levels**: -15dB (very far) to +5dB (very close)
- **Background Noises**: 1-3 sources combined
- **Purpose**: Simulate distance and environmental conditions

### No-Drone Augmentation (Class 0) ✨ NEW
- **Method**: Complex background mixing with audio effects
- **Effects**:
  - Pitch shifting: ±2 semitones
  - Time stretching: 0.9x - 1.1x speed
  - Amplitude variations
  - Multi-source mixing
- **Purpose**: Create diverse non-drone soundscapes

## 🔍 Dataset Splitting

### Split Strategy
- **Train (70%)**: For model learning
- **Validation (15%)**: For hyperparameter tuning
- **Test (15%)**: Final evaluation only

### Zero Leakage Guarantee
- Each file appears in ONLY ONE split
- Class balance maintained across all splits
- Verification included in `split_dataset.py`
- Split indices saved in `split_info.json`

## 📈 Expected Volumes

### Default Configuration
```
Original DADS:        100 per class = 200 total
Augmented Drones:     200 total (5 SNR categories)
Augmented No-Drones:  200 total (4 complexity categories)
Combined:             300 per class = 600 total

After split (70/15/15):
- Train:      420 samples (210 per class)
- Validation:  90 samples (45 per class)
- Test:        90 samples (45 per class)
```

## 🔧 Usage with Existing Scripts

### Folders 1-5 remain UNCHANGED!

The existing training and evaluation scripts work automatically through `config.py`:

```python
# In config.py (already configured)
DATASET_NAME = os.environ.get("DATASET_ROOT_OVERRIDE", "dataset_combined")
DATASET_ROOT = PROJECT_ROOT / "0 - DADS dataset extraction" / DATASET_NAME
```

### Switching Datasets

```bash
# Use training dataset
set DATASET_ROOT_OVERRIDE=dataset_train
python "1 - Preprocessing and Features Extraction\Mel_Preprocess_and_Feature_Extract.py"
python "2 - Model Training\CNN_Trainer.py"

# Use test dataset (for evaluation)
set DATASET_ROOT_OVERRIDE=dataset_test_final
python "3 - Single Model Performance Calculation\CNN_and_CRNN_Performance_Calcs.py"
```

## ✅ Validation Checklist

After running the pipeline, verify:

- [ ] `dataset_train/` has ~70% of combined dataset
- [ ] `dataset_val/` has ~15% of combined dataset
- [ ] `dataset_test_final/` has ~15% of combined dataset
- [ ] Both classes (0/ and 1/) present in all splits
- [ ] `split_info.json` shows zero overlap
- [ ] `augmentation_metadata.json` exists in dataset_augmented/
- [ ] Class balance: approximately 50/50 in each split

## 🐛 Troubleshooting

### "No class directories found"
- Ensure source dataset has `0/` and `1/` subdirectories
- Check file permissions

### "Ratios must sum to 1.0"
- Verify train + val + test = 1.0
- Use `--no-val` flag if skipping validation

### "Librosa effects not working"
- Install latest librosa: `pip install librosa>=0.10.0`
- Check that numba is installed: `pip install numba`

## 📚 References

- **DADS Dataset**: https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples
- **Librosa Effects**: https://librosa.org/doc/main/generated/librosa.effects.html
- **Audio Augmentation**: Standard practices for speech/audio recognition

## 🔄 Version History

### v2.0 (Current)
- ✨ Added no-drone augmentation
- ✨ Train/val/test splitting
- ✨ Audio effects (pitch/time)
- ✨ Automated pipeline
- ✨ Zero leakage verification

### v1.0 (Legacy)
- Drone-only augmentation
- Manual dataset management
- Single combined dataset

---

**Questions?** Check `README.md` in project root or review inline documentation in scripts.
