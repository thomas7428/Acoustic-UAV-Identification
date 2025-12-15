"""
Regenerate a test-time MEL index (one MEL per WAV) using centralized config parameters.

This script walks `config.DATASET_ROOT`, computes a single MEL per audio file
using `config.MEL_DURATION`, `config.MEL_N_MELS`, etc., pads/truncates to
`config.MEL_TIME_FRAMES`, and writes `mel_test_index.json` next to the
original features file.

Usage:
    .venv/bin/python3 "1 - Preprocessing and Features Extraction/regenerate_mel_test_index_from_wavs.py"
"""
import json
import os
from pathlib import Path
import sys
import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

FEATURES_DIR = Path(config.EXTRACTED_FEATURES_DIR)
OUT_PATH = FEATURES_DIR / "mel_test_index.json"
DATASET_ROOT = Path(config.DATASET_ROOT)


def collect_file_list(dataset_root):
    files = []
    for dirpath, dirnames, filenames in os.walk(dataset_root):
        if Path(dirpath) == Path(dataset_root):
            continue
        for f in sorted(filenames):
            if f.lower().endswith('.wav'):
                files.append(os.path.join(dirpath, f))
    return files


def infer_label_from_path(path):
    # Expect directory structure .../<category>/<wavfile.wav> where category is '0' or '1'
    p = Path(path)
    parent = p.parent.name
    try:
        return int(parent)
    except Exception:
        return None


def build_index():
    print("Regenerating mel_test_index.json from WAVs using config settings...")
    files = collect_file_list(DATASET_ROOT)
    print(f"Found {len(files)} WAV files under {DATASET_ROOT}")

    if len(files) == 0:
        print("No files found — aborting")
        return 1

    names = []
    mels = []
    labels = []
    mapping = []

    for f in files:
        basename = os.path.basename(f)
        try:
            audio, sr = librosa.load(f, sr=config.SAMPLE_RATE, duration=config.MEL_DURATION)
            mel = librosa.feature.melspectrogram(y=audio,
                                                 sr=sr,
                                                 n_mels=config.MEL_N_MELS,
                                                 n_fft=config.MEL_N_FFT,
                                                 hop_length=config.MEL_HOP_LENGTH)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            # Ensure shape is (n_mels, time)
            if mel_db.shape[0] != config.MEL_N_MELS:
                mel_db = mel_db.reshape((config.MEL_N_MELS, -1))

            # Pad / truncate to exact time frames
            if mel_db.shape[1] < config.MEL_TIME_FRAMES:
                pad_width = config.MEL_TIME_FRAMES - mel_db.shape[1]
                mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=(config.MEL_PAD_VALUE,))
            else:
                mel_db = mel_db[:, :config.MEL_TIME_FRAMES]

            names.append(basename)
            mels.append(mel_db.tolist())
            labels.append(infer_label_from_path(f))
        except Exception as e:
            print(f"[WARN] Failed to process {f}: {e}")

    out = {
        'mapping': mapping,
        'names': names,
        'mel': mels,
        'labels': labels
    }

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as fp:
        json.dump(out, fp)

    print(f"[OK] Wrote {OUT_PATH} with {len(names)} entries")
    return 0


if __name__ == '__main__':
    sys.exit(build_index())
