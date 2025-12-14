# 🧪 Testing Framework

This folder contains the comprehensive test suite for the Acoustic UAV Identification project.

## 📁 Test Files

### `test_augmentation.py`
Tests for advanced audio augmentation functions:
- **Doppler shift** - Pitch shift simulation for drone movement
- **Intensity modulation** - Sinusoidal envelope for motor variations
- **Reverberation** - Echo effects for environment reflections
- **Time stretching** - Rotor speed change simulation

Test categories:
- ✅ Output shape validation
- ✅ Amplitude range checks
- ✅ Sequential augmentation pipelines
- ✅ Edge cases (empty, short, zero signals)
- ✅ Dtype preservation

### `test_loss_functions.py`
Tests for custom loss functions and metrics:
- **Focal loss** - Class imbalance and hard example handling
- **Weighted BCE** - Class weighting for imbalanced datasets
- **Loss factory** - `get_loss_function()` with different configurations
- **Metrics** - Accuracy, precision, recall, AUC, TP/FP/TN/FN

Test categories:
- ✅ Loss output validation (shape, positivity)
- ✅ Parameter effects (alpha, gamma, class weights)
- ✅ Perfect vs wrong predictions
- ✅ Numerical stability (extreme values)
- ✅ TensorFlow compatibility

### `test_spec_augment.py`
Tests for SpecAugment implementation:
- **Time masking** - Temporal occlusion
- **Frequency masking** - Spectral occlusion
- **Combined masking** - Both time and frequency

Test categories:
- ✅ Masking application verification
- ✅ Mask width constraints
- ✅ Number of masks effect
- ✅ Randomness testing
- ✅ Edge cases (small spectrograms, single bins)
- ✅ Integration with realistic mel spectrograms

## 🚀 Running Tests

### Quick Run (All Tests)
```bash
cd "7 - Tests"
./run_tests.sh
```

### Run with Coverage
```bash
./run_tests.sh --coverage
```

### Run Specific Test Category
```bash
# Only augmentation tests
python3 test_augmentation.py

# Only loss function tests
python3 test_loss_functions.py

# Only SpecAugment tests
python3 test_spec_augment.py
```

### Using pytest (if installed)
```bash
# All tests with verbose output
pytest -v

# Specific test file
pytest test_augmentation.py -v

# With coverage
pytest --cov --cov-report=html

# Specific test class
pytest test_augmentation.py::TestDopplerShift -v

# Specific test method
pytest test_augmentation.py::TestDopplerShift::test_doppler_shift_shape -v
```

## 📊 Coverage Goals

Target: **>80% code coverage** for Phase 1 components

Current coverage areas:
- ✅ Audio augmentation functions (`augment_dataset_v2.py`)
- ✅ Loss functions (`loss_functions.py`)
- ✅ SpecAugment (`Mel_Preprocess_and_Feature_Extract.py`)

## 🔧 Dependencies

Required packages:
```bash
pip install numpy tensorflow librosa scipy
```

Optional (for better test reports):
```bash
pip install pytest pytest-cov
```

## 📝 Test Structure

Each test file follows this structure:

1. **Setup** - Create test data and fixtures
2. **Test cases** - Individual test methods
3. **Edge cases** - Boundary conditions and error handling
4. **Integration tests** - Multi-component interactions
5. **Summary** - Colored output with pass/fail counts

## ✅ Success Criteria

Phase 1 testing requirements:
- ✅ All unit tests pass
- ✅ >80% code coverage
- ✅ Edge cases handled gracefully
- ✅ No memory leaks or numerical instabilities
- ✅ Integration tests verify end-to-end pipelines

## 🐛 Debugging Failed Tests

If tests fail:

1. **Read the error message** - Shows which assertion failed
2. **Check test output** - Verbose mode shows intermediate values
3. **Run single test** - Isolate the failing test
4. **Add print statements** - Debug specific values
5. **Check dependencies** - Ensure all packages installed correctly

Example:
```bash
# Run single test with extra verbosity
pytest test_augmentation.py::TestDopplerShift::test_doppler_shift_shape -vv
```

## 📈 Future Test Additions

Planned for Phase 2:
- Multi-scale segmentation tests
- Mixup augmentation tests
- Weighted ensemble voting tests

Planned for Phase 3:
- Attention mechanism tests
- Temporal coherence loss tests
- Lightweight model tests (MobileNet, TCN)

## 🎯 Integration with Pipeline

These tests are automatically run by `run_full_pipeline.sh` when using the `--test` flag:

```bash
cd ..
./run_full_pipeline.sh --test
```

This ensures all components are validated before training begins.
