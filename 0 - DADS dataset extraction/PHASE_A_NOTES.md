# Phase A Notes — 0 - DADS dataset extraction

Date: 2025-12-17

This file summarizes the Phase A work applied to the DADS extraction area.

Changes applied:

- Centralized audio parameters in `config.py` (SAMPLE_RATE, AUDIO_DURATION_S, MEL_* params, and `AUDIO_WAV_SUBTYPE`).
- `augment_dataset_v2.py` hardened:
  - Now requires `--config` (no silent fallbacks).
  - Uses `project_config.SAMPLE_RATE` and `project_config.AUDIO_DURATION_S` as single source of truth.
  - Forces exact sample durations (looping with crossfade / trimming) before saving.
  - Writes WAV files using `project_config.AUDIO_WAV_SUBTYPE`.
- `download_and_prepare_dads.py` updated:
  - Converts multi-channel audio to mono by averaging channels.
  - Ensures `float32` dtype and resamples to project sample rate.
  - Uses `config.AUDIO_WAV_SUBTYPE` when writing WAVs.
- New helper tools under `tools/`:
  - `tools/organize_datasets.py` — create canonical dataset directories and helpers to import/split.
  - `tools/validate_dataset.py` — scan WAVs and precomputed feature JSONs for consistency issues.
  - `tools/archive.py` — safe archiving with `--dry-run` and explicit `--confirm` before cleanup.

Outstanding decisions / recommendations:

- MEL frames consistency: `config.MEL_TIME_FRAMES` is 90, but existing precomputed MEL JSONs use 173 frames. Choose one:
  1. Re-extract MEL features for all datasets to match `MEL_TIME_FRAMES=90` (recommended for consistency), or
  2. Keep existing precomputed features (173 frames) and make trainers always derive input shape from the loaded data (already partly implemented).

- Refactor `master_setup_v2.py` to use the `tools/` helpers (archive, organize, augment) for safer and clearer orchestration.

- Once you decide on MEL frames, run a full small-scale extraction + preprocessing + train cycle to verify accuracy improvements.

Quick commands used during Phase A:

```bash
# Archive dry-run
python3 tools/archive.py --out ./archives --dry-run

# Augmentation dry-run
python3 "0 - DADS dataset extraction/augment_dataset_v2.py" --config augment_config_v2.json --dry-run

# Small DADS extraction (dev)
python3 "0 - DADS dataset extraction/download_and_prepare_dads.py" --output dataset_dads_dev --max-per-class 100
```

If you'd like I can now:

- Re-extract MEL features to the canonical `MEL_TIME_FRAMES` (this will take time), or
- Refactor `master_setup_v2.py` to call the `tools/` helpers, or
- Run a small verification pipeline (download small DADS subset → augment → extract features → quick train) to verify behavior.
