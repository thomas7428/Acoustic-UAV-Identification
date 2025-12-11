# Migration Guide: v1 → v2 Enhanced Dataset Pipeline

## 📋 Overview

This guide helps you transition from the original dataset pipeline to the enhanced v2.0 system with balanced augmentation and proper train/test splitting.

## 🔄 What Changed?

### v1.0 (Original)
```
dataset_test/          # Small test set (2 files)
dataset_augmented/     # Only drones augmented
dataset_combined/      # Mixed for both train AND test ❌ LEAKAGE
```

### v2.0 (Enhanced)
```
dataset_test/          # Original DADS samples
dataset_augmented/     # BOTH classes augmented ✨
dataset_combined/      # Temporary (originals + augmented)
dataset_train/         # 70% - Training only ✅
dataset_val/           # 15% - Validation only ✅
dataset_test/          # 15% - Testing only ✅ ZERO OVERLAP
```

## 🚀 Quick Migration

### Option A: Fresh Start (Recommended)

```bash
cd "0 - DADS dataset extraction"

# Run the new automated pipeline
python master_setup_v2.py --drone-samples 100 --no-drone-samples 100

# That's it! Everything is set up with proper splits
```

### Option B: Use Existing Data

If you already have `dataset_combined/`, you can skip download and augmentation:

```bash
# Just split your existing combined dataset
python split_dataset.py --source dataset_combined --train 0.7 --val 0.15 --test 0.15

# Extract features from training set
set DATASET_ROOT_OVERRIDE=dataset_train
python "..\1 - Preprocessing and Features Extraction\Mel_Preprocess_and_Feature_Extract.py"
```

## 📊 Comparison Table

| Feature | v1.0 | v2.0 |
|---------|------|------|
| Drone augmentation | ✅ Yes | ✅ Yes |
| No-drone augmentation | ❌ No | ✅ Yes |
| Train/test split | ❌ No separation | ✅ Proper split |
| Data leakage | ⚠️ High risk | ✅ Zero risk |
| Audio effects | ❌ Limited | ✅ Pitch shift, time stretch |
| Validation set | ❌ No | ✅ Yes |
| Reproducibility | ⚠️ Partial | ✅ Full (random seed) |
| Verification | ❌ No | ✅ Automatic |

## 🔧 Updating Your Workflow

### Training Scripts (Folders 2-4)

**No changes needed!** Scripts automatically use `config.py`:

```python
# config.py handles everything
DATASET_NAME = os.environ.get("DATASET_ROOT_OVERRIDE", "dataset_combined")
```

**Before training:**
```bash
# OLD way (v1.0)
python "2 - Model Training\CNN_Trainer.py"  # Uses dataset_combined ❌

# NEW way (v2.0)
set DATASET_ROOT_OVERRIDE=dataset_train
python "2 - Model Training\CNN_Trainer.py"  # Uses dataset_train ✅
```

### Performance Calculation Scripts (Folder 3)

**Before testing:**
```bash
# OLD way (v1.0)
python "3 - Single Model Performance Calculation\CNN_and_CRNN_Performance_Calcs.py"  # Uses dataset_combined ❌ SAME AS TRAINING!

# NEW way (v2.0)
set DATASET_ROOT_OVERRIDE=dataset_test
python "3 - Single Model Performance Calculation\CNN_and_CRNN_Performance_Calcs.py"  # Uses dataset_test ✅ NEVER SEEN IN TRAINING!
```

### Visualization Scripts (Folder 6)

Update `performance_by_distance.py` to use test dataset:

```python
# Line ~151 in performance_by_distance.py
# OLD:
combined_path = config.PROJECT_ROOT / "0 - DADS dataset extraction" / "dataset_combined"

# NEW:
combined_path = config.PROJECT_ROOT / "0 - DADS dataset extraction" / "dataset_test"
```

Or set environment variable before running:
```bash
set DATASET_ROOT_OVERRIDE=dataset_test
python "6 - Visualization\performance_by_distance.py"
```

## 📝 Script Equivalence

| v1.0 Script | v2.0 Script | Notes |
|-------------|-------------|-------|
| `download_and_prepare_dads.py` | Same | ✅ Compatible |
| `augment_dataset.py` | `augment_dataset_v2.py` | ✨ Enhanced |
| `augment_config.json` | `augment_config_v2.json` | ✨ Enhanced |
| N/A | `split_dataset.py` | ✨ NEW |
| `master_setup.py` | `master_setup_v2.py` | ✨ Enhanced |

## ✅ Verification Checklist

After migration, verify:

- [ ] `dataset_train/` exists with ~70% of data
- [ ] `dataset_val/` exists with ~15% of data
- [ ] `dataset_test/` exists with ~15% of data
- [ ] `split_info.json` shows zero overlap between splits
- [ ] Both classes (0/ and 1/) are balanced in all splits
- [ ] `augmentation_metadata.json` shows both drone AND no-drone augmentation
- [ ] Environment variable works: `set DATASET_ROOT_OVERRIDE=dataset_train`
- [ ] Features extracted from `dataset_train` only

## 🐛 Common Issues

### "DATASET_ROOT_OVERRIDE not working"

**Windows (PowerShell):**
```powershell
$env:DATASET_ROOT_OVERRIDE = "dataset_train"
python script.py
```

**Windows (CMD):**
```cmd
set DATASET_ROOT_OVERRIDE=dataset_train
python script.py
```

**Linux/Mac:**
```bash
export DATASET_ROOT_OVERRIDE=dataset_train
python script.py
```

### "Models still show 100% accuracy"

This was caused by testing on training data. With v2.0:
1. Ensure you're using `dataset_test` for evaluation
2. Check `split_info.json` to verify zero overlap
3. Re-train models on `dataset_train` only

### "Not enough samples in test set"

Increase download count:
```bash
python master_setup_v2.py --drone-samples 500 --no-drone-samples 500
```

## 📈 Expected Performance Changes

### Before (v1.0 - Data Leakage)
```
Training on dataset_combined:   57% accuracy
Testing on dataset_combined:    100% accuracy ❌ OVERFITTING
```

### After (v2.0 - Proper Split)
```
Training on dataset_train:      ~60% accuracy
Testing on dataset_test:        ~55-60% accuracy ✅ REALISTIC
```

The test accuracy may actually be **lower** than before, but that's correct! It means:
- No data leakage
- True generalization performance
- Reliable evaluation

## 🔄 Rollback Plan

If you need to revert to v1.0:

1. Keep old scripts (they're still in the directory)
2. Use `augment_config.json` (not v2)
3. Use `augment_dataset.py` (not v2)
4. Don't set `DATASET_ROOT_OVERRIDE`
5. Use `dataset_combined` as before

**However, we recommend v2.0 for production!**

## 📚 Additional Resources

- `README_V2.md` - Complete v2.0 documentation
- `augment_config_v2.json` - Configuration reference
- `split_info.json` - Split verification data
- `augmentation_metadata.json` - Augmentation details

## ❓ FAQ

**Q: Can I use v1 and v2 simultaneously?**  
A: Yes! v2 scripts have `_v2` suffix and use different directories.

**Q: Do I need to retrain my models?**  
A: Yes, to get reliable results. Old models trained on `dataset_combined` suffer from data leakage.

**Q: What about my existing trained models?**  
A: Keep them for comparison, but train new ones on `dataset_train` for proper evaluation.

**Q: Is the validation set mandatory?**  
A: No, you can use `--no-val` flag in `split_dataset.py` or `master_setup_v2.py` for a simple 80/20 train/test split.

**Q: Can I adjust split ratios?**  
A: Yes! Use `--train`, `--val`, `--test` parameters. Common: 80/10/10, 70/15/15, 60/20/20.

---

**Need help?** Check the detailed documentation in `README_V2.md` or review the inline comments in the scripts.
