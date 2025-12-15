Deprecated tools
================

This folder contains utilities that were used during development but are no
longer part of the canonical pipeline. Files are preserved for auditability and
reproducibility, but are not run by default.

Why deprecated
--------------
- The project now enforces a single canonical precomputed MEL-per-WAV workflow
  produced by `1 - Preprocessing and Features Extraction/regenerate_mel_test_index_from_wavs.py`.
- Older scripts that rebuilt `mel_test_index.json` from segmented features were
  brittle (depended on traversal order and segment counts) and caused mismatches
  between training and test-time feature shapes.

What to use instead
-------------------
- Use `1 - Preprocessing and Features Extraction/regenerate_mel_test_index_from_wavs.py`
  to regenerate a consistent `mel_test_index.json` (one MEL per WAV) that matches
  training parameters defined in `config.py`.

If you need to re-enable an old tool temporarily, copy it out of this folder to
its original location, and inspect the header comment to understand limitations.