# Model Pool Documentation

## Unified Configuration

This deployment system now uses **a single configuration file** (`deployment_config.json`) that supports **8 models** (4 original + 4 new).

### Quick Start

**Use NEW models (recommended):**
```json
"enabled_models": ["new_CNN", "new_RNN", "new_CRNN", "new_Attention-CRNN"]
```

**Use ORIGINAL models:**
```json
"enabled_models": ["CNN", "RNN", "CRNN", "Attention-CRNN"]
```

**Mix both pools:**
```json
"enabled_models": ["new_CNN", "CNN", "new_Attention-CRNN", "CRNN"]
```

## Dynamic Threshold Loading

Thresholds are **loaded automatically** from the calibration file:
```json
"thresholds_file": "../0 - DADS dataset extraction/results/calibrated_thresholds.json"
```

**Real-time threshold updates:**
- Set `"reload_thresholds_every_prediction": true` to reload thresholds on every prediction
- Modify the calibration JSON file and the detector will use new values instantly
- No need to restart the system!

## Available Models (8 Total)

### Original Pool (4 models)
- `CNN`: cnn_model.keras (4.0 MB)
- `RNN`: rnn_model.keras (8.4 MB)
- `CRNN`: crnn_model.keras (2.4 MB)
- `Attention-CRNN`: attention_crnn_model.keras (15 MB)

### New Pool (Jan 10-11, 2026) - 4 models
- `new_CNN`: new_cnn_model.keras (4.0 MB) - **F1=0.92, Acc=91.5%**
- `new_RNN`: new_rnn_model.keras (8.4 MB) - **F1=0.84, Acc=77.2%**
- `new_CRNN`: new_crnn_model.keras (2.4 MB) - **F1=0.91, Acc=90.5%**
- `new_Attention-CRNN`: new_attention_crnn_model.keras (15 MB) - **F1=0.93, Acc=91.4%**

**New models improvements:**
- Trained with optimized batch size (128)
- Better CPU threading (intra=12, inter=4)
- RNN threshold fixed (0.51 → 0.38)
- All thresholds properly calibrated with balanced precision

## Configuration File Structure

```json
{
  "detection": {
    "thresholds_file": "../0 - DADS dataset extraction/results/calibrated_thresholds.json",
    "enabled_models": ["new_CNN", "new_RNN", "new_CRNN", "new_Attention-CRNN"],
    "reload_thresholds_every_prediction": true
  },
  "models": {
    "available_models": {
      "new_CNN": {
        "filename": "new_cnn_model.keras",
        "calibrated_key": "CNN"
      }
    }
  }
}
```

**Key fields:**
- `enabled_models`: List of models to load (choose from 8 available)
- `thresholds_file`: Path to calibration JSON
- `reload_thresholds_every_prediction`: Enable real-time threshold updates
- `calibrated_key`: Maps model to threshold in calibration file

## Performance Comparison

| Model | Original Threshold | New Threshold | New F1-Score | New Accuracy |
|-------|-------------------|---------------|--------------|--------------|
| CNN | 0.38 → 0.40 | 0.40 | 0.92 | 91.5% |
| RNN | 0.51 → 0.38 | 0.38 | 0.84 | 77.2% |
| CRNN | 0.40 → 0.37 | 0.37 | 0.91 | 90.5% |
| Attention-CRNN | 0.42 (same) | 0.42 | 0.93 | 91.4% |

**Key Improvements:**
- RNN threshold significantly reduced (0.51 → 0.38), fixing the "predict all as drone" bug
- All models properly calibrated with balanced precision constraints
- Better handling of distance-specific performance

## Model Files Size

| Model | Size | Description |
|-------|------|-------------|
| CNN | 4.0 MB | Fastest inference, good baseline |
| RNN | 8.4 MB | Temporal patterns, recurrent layers |
| CRNN | 2.4 MB | Smallest, balanced CNN+RNN hybrid |
| Attention-CRNN | 15 MB | Best performance, attention mechanism |

## Calibration Details

All thresholds were calibrated using **hierarchical multi-criteria optimization**:

**Tier 1 (Hard Constraints):**
- Min Recall (Sensitivity): 90%
- Min Precision Drone (PPV): 70%
- Min Precision Ambient (NPV): 85%

**Tier 2 (Optimization Target):**
- Maximize: balanced_precision = min(PPV, NPV)

**Tier 3 (Tie-breakers):**
- F1-score → Recall → Lower threshold

## Recommendations

For **best overall performance**: Use `new_attention_crnn_model.keras` (F1=0.93)

For **fastest inference**: Use `new_cnn_model.keras` (F1=0.92, smallest latency)

For **balanced resource usage**: Use `new_crnn_model.keras` (F1=0.91, only 2.4 MB)

For **ensemble/voting**: Use all 4 models with `deployment_config_new_models.json`

## Deployment on Raspberry Pi

1. **Copy entire directory** to Raspberry Pi:
   ```bash
   scp -r . pi@raspberrypi:/home/pi/drone-detection/
   ```

2. **On Raspberry Pi**, choose configuration:
   ```bash
   # Use new models (recommended)
   python3 drone_detector.py --config deployment_config_new_models.json --continuous
   
   # Or use original models for comparison
   python3 drone_detector.py --config deployment_config.json --continuous
   ```

3. **Test both configurations** to compare performance on real-world data

## Notes

- Both model pools share the same feature extraction pipeline (MEL spectrogram, 44 bands)
- No normalization applied to preserve SNR differences across distances
- All models expect input shape: (1, 44, 90, 1)
- Audio: 22050 Hz, 4 seconds duration
- Output: Single probability (sigmoid activation) for drone class
