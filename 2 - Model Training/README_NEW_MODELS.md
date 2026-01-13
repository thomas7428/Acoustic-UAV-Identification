# New SOTA Model Architectures (Jan 2025)

This directory contains 8 model trainers for acoustic UAV detection:

## Original Models (4)
1. **CNN** - Baseline convolutional neural network
2. **RNN** - Recurrent network with LSTM/BiLSTM layers
3. **CRNN** - Combined CNN + RNN architecture
4. **Attention-CRNN** - CRNN enhanced with attention mechanisms

## New SOTA Models (4)
Added on January 13, 2025 to improve performance beyond F1~0.93

### 5. EfficientNet-B0
- **Architecture**: Mobile Inverted Bottleneck Convolution (MBConv) with Squeeze-and-Excitation
- **Key Features**:
  - Compound scaling (depth × width × resolution)
  - SE blocks for channel-wise attention
  - Swish activation function
  - 7 stages with varying kernel sizes (3x3, 5x5)
- **Expected Performance**: F1 ~0.96, Size: 5 MB, Inference: 90ms
- **Best for**: High accuracy applications

### 6. MobileNet-Audio
- **Architecture**: Depthwise separable convolutions
- **Key Features**:
  - 13 depthwise separable blocks
  - Ultra-lightweight design
  - 5 stages with progressive downsampling
  - Optimized for embedded devices
- **Expected Performance**: F1 ~0.93-0.95, Size: 1-2 MB, Inference: 18-25ms
- **Best for**: Raspberry Pi deployment, real-time inference

### 7. Conformer
- **Architecture**: CNN + Transformer hybrid
- **Key Features**:
  - Multi-Head Self-Attention (4 heads)
  - Depthwise convolution module (kernel=31)
  - GLU (Gated Linear Units) activation
  - Macaron-style Feed-Forward architecture
  - 4-6 Conformer blocks
- **Expected Performance**: F1 ~0.96-0.98, Size: 8 MB, Inference: 130ms
- **Best for**: State-of-the-art accuracy (SOTA)

### 8. TCN (Temporal Convolutional Network)
- **Architecture**: Dilated causal convolutions
- **Key Features**:
  - Exponentially growing receptive field (dilation: 1, 2, 4, 8, 16, 32)
  - Residual connections between blocks
  - GlobalAveragePooling1D
  - Replaces slow RNN with faster temporal processing
- **Expected Performance**: F1 ~0.88-0.92, Size: 3 MB, Inference: 40ms
- **Best for**: Faster alternative to RNN

## Training Parameters
All trainers support:
- **Loss functions**: bce, focal, class_balanced, distance_weighted
- **Batch size**: Configurable via `config.BATCH_SIZE` (default: 128)
- **Learning rate**: Configurable via `config.LEARNING_RATE` (default: 0.0001)
- **Early stopping**: Patience 10 epochs
- **LR reduction**: Factor 0.5, patience 5 epochs

## Usage

### Train single model:
```bash
cd "2 - Model Training"
python EfficientNet_Trainer.py --loss focal --epochs 50
python MobileNet_Trainer.py --loss bce --epochs 50
python Conformer_Trainer.py --loss focal --epochs 50 --blocks 4
python TCN_Trainer.py --loss bce --epochs 50
```

### Train all models via pipeline:
```bash
./run_full_pipeline.sh --parallel
```

### Train specific models only:
```bash
./run_full_pipeline.sh --models EFFICIENTNET,MOBILENET,CONFORMER,TCN
```

## Input/Output

### Input Shape
- **Conv2D models** (CNN, CRNN, Attention-CRNN, EfficientNet, MobileNet): `(44, 90, 1)`
- **1D models** (RNN, TCN, Conformer): `(90, 44)` - reshaped internally

### Output Files
Each trainer creates:
- `{model}_model.keras` - Saved model
- `{model}_history.csv` - Training history
- `{model}_accuracy.json` - Test accuracy metrics

## Integration
All models are integrated into:
- `run_full_pipeline.sh` - Training pipeline
- `Universal_Perf_Tester.py` - Testing
- `calibrate_thresholds.py` - Threshold calibration
- `config.py` - Model paths configuration

## Performance Comparison (Expected)

| Model | F1 Score | Size | Inference Time | Best Use Case |
|-------|----------|------|----------------|---------------|
| CNN | 0.92 | 4 MB | 50ms | Baseline |
| RNN | 0.84 | 8.4 MB | 120ms | Temporal patterns |
| CRNN | 0.91 | 2.4 MB | 60ms | Balanced |
| Attention-CRNN | 0.93 | 15 MB | 85ms | Current best |
| **EfficientNet** | **0.96** | 5 MB | 90ms | **Highest accuracy** |
| **MobileNet** | 0.94 | 2 MB | 25ms | **Fastest** |
| **Conformer** | **0.97** | 8 MB | 130ms | **SOTA** |
| **TCN** | 0.90 | 3 MB | 40ms | Fast temporal |

## Dependencies
All trainers use:
- TensorFlow/Keras
- NumPy
- pandas
- scikit-learn
- Feature loader from `tools/feature_loader.py`
- Loss functions from `loss_functions.py`
