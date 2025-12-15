"""
build_mel_test_index.py (deprecated copy)

This is the deprecated copy of the utility that built `mel_test_index.json`
from a segmented features JSON. It was moved to `tools/deprecated/` because
we now use a single canonical regenerating script that creates exactly one
MEL per WAV using the trainer parameters: 

`1 - Preprocessing and Features Extraction/regenerate_mel_test_index_from_wavs.py`

Keep this file for traceability only — do NOT run it in the canonical
pipeline. See `README.md` in this folder for details.
"""

# The file below is the original script content (kept for auditability).

import json
import os
from pathlib import Path
import sys

# Import project config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

FEATURES_PATH = Path(config.MEL_TRAIN_PATH)
OUT_PATH = FEATURES_PATH.parent / "mel_test_index.json"
DATASET_ROOT = Path(config.DATASET_ROOT)


def collect_file_list(dataset_root):
    """Replicate the traversal order used in feature extraction (os.walk without sorting)."""
    files = []
    for dirpath, dirnames, filenames in os.walk(dataset_root):
        # Skip root directory entries (the extractor only processed subfolders)
        if Path(dirpath) == Path(dataset_root):
            continue
        for f in filenames:
            if f.lower().endswith('.wav'):
                files.append(os.path.join(dirpath, f))
    return files


def main():
    print("Building mel_test_index.json from precomputed features...")

    if not FEATURES_PATH.exists():
        print(f"[ERROR] Features file not found: {FEATURES_PATH}")
        return 1

    # Load features json (may be large)
    print(f"Loading features JSON: {FEATURES_PATH} (this may take a while)")
    with open(FEATURES_PATH, 'r') as f:
        data = json.load(f)

    mel = data.get('mel', [])
    labels = data.get('labels', [])

    if not mel:
        print("[ERROR] No 'mel' array found in features file")
        return 1

    print(f"Total mel vectors: {len(mel)}")

    # Collect file list
    files = collect_file_list(DATASET_ROOT)
    print(f"Files found in dataset: {len(files)}")

    if len(files) == 0:
        print("[ERROR] No audio files found under dataset root: {DATASET_ROOT}")
        return 1

    # Infer num_segments
    if len(mel) % len(files) != 0:
        print("[ERROR] mel length is not a multiple of file count. Cannot safely reconstruct index.")
        print(f"mel len: {len(mel)}, files: {len(files)}")
        return 1

    num_segments = len(mel) // len(files)
    print(f"Inferred num_segments per file: {num_segments}")

    # Build test index by picking the best valid segment per file.
    # Strategy:
    #  - Discard segments that are mostly sentinel values (-100.0). A segment with more than
    #    `max_sentinel_fraction` of its elements equal to -100 is considered invalid.
    #  - Among valid segments choose the one with highest mean energy (dB). If no valid
    #    segments exist for a file, fall back to choosing the segment with highest mean
    #    (same as before) but emit a warning — this is a degraded case.
    out_names = []
    out_mel = []
    out_labels = []

    import numpy as _np
    max_sentinel_fraction = 0.20  # allow up to 20% of values being -100 in a segment

    for i, fname in enumerate(files):
        # compute the segment range for this file
        start = i * num_segments
        end = start + num_segments
        segs = mel[start:end]
        seg_labels = labels[start:end]

        # Convert to numpy arrays for numeric ops
        segs_arr = [_np.array(s, dtype=float) for s in segs]

        # Compute sentinel fraction and mean energy
        sentinel_fracs = [(_np.sum(s == -100.0) / float(s.size)) for s in segs_arr]
        means = [_np.mean(s) for s in segs_arr]

        # Select valid indices where sentinel fraction <= threshold
        valid_indices = [j for j, frac in enumerate(sentinel_fracs) if frac <= max_sentinel_fraction]

        if valid_indices:
            # choose best among valid segments
            valid_means = [_np.mean(segs_arr[j]) for j in valid_indices]
            best_rel = int(_np.argmax(valid_means))
            best_idx = valid_indices[best_rel]
        else:
            # Fallback: choose the segment with highest mean (no valid segments found)
            best_idx = int(_np.argmax(means))
            print(f"[WARNING] File {os.path.basename(fname)}: no valid segments under sentinel threshold; falling back to highest-mean segment")

        chosen = segs_arr[best_idx]
        chosen_label = int(seg_labels[best_idx]) if seg_labels is not None else None

        # Ensure orientation is (n_mels, time). Some extractors saved as (time, n_mels).
        # Heuristic: if chosen.shape[0] > chosen.shape[1], assume (time, n_mels) and transpose.
        if chosen.shape[0] > chosen.shape[1]:
            chosen = chosen.T

        out_names.append(os.path.basename(fname))
        out_mel.append(chosen.tolist())
        out_labels.append(chosen_label)

    out = {
        'mapping': data.get('mapping', []),
        'names': out_names,
        'mel': out_mel,
        'labels': out_labels
    }

    print(f"Writing test index to: {OUT_PATH}")
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f)

    print("[OK] mel_test_index.json created")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
