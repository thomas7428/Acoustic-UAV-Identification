#!/usr/bin/env python3
"""Compare a precomputed MEL (from mel_test_index.json) with on-the-fly MEL computed like trainers.
Prints summary statistics and saves a small PNG heatmap for visual inspection.
"""
import sys
audio, sr = librosa.load(str(sample_path), sr=22050, duration=4.0)
diff = mel_db - pre_mel_t_norm
out = Path('compare_mel_sample.png')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import json
import numpy as np
import config

# This script now only INSPECTS precomputed MELs. On-the-fly computation is disabled.
TEST_INDEX = Path(config.EXTRACTED_FEATURES_DIR) / "mel_test_index.json"

if not TEST_INDEX.exists():
    print("mel_test_index.json not found; run build_mel_test_index.py first")
    sys.exit(1)

with open(TEST_INDEX, 'r') as f:
    idx = json.load(f)

names = idx.get('names', [])
mels = idx.get('mel', [])

if not names:
    print('No names in test index')
    sys.exit(1)

# pick first entry
sample_name = names[0]
pre_mel = np.array(mels[0])
print('Sample from index:', sample_name)
print('precomputed MEL shape:', pre_mel.shape)
print('precomputed stats: min', pre_mel.min(), 'max', pre_mel.max(), 'mean', pre_mel.mean())
out = Path('compare_precomputed_summary.txt')
out.write_text(f"name: {sample_name}\nshape: {pre_mel.shape}\nmin: {pre_mel.min()}\nmax: {pre_mel.max()}\nmean: {pre_mel.mean()}\n")
print('Wrote summary to', out)
