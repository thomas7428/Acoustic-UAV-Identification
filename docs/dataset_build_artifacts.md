### Dataset build artifacts (v4)

This file documents the minimal metadata written by the v4 build wrapper (`tools/build_dataset_v4.sh`).

Files created at `datasets/<dataset_id>/`:

- `effective_config.json` — exact JSON config passed to `augment_dataset_v4.py` (post-merge/overrides).
- `build_info.json` — short build metadata, contains:
  - `dataset_id`
  - `git_sha` (from `git rev-parse HEAD`)
  - `timestamp_utc`
  - `command` (argv string used to invoke the build)
  - `env` (recorded relevant env vars used for determinism)
  - `sha256` (hashes of `effective_config.json` and `splits.json`)

Rationale:

- Keep a single source-of-truth for reproducibility and auditing.
- Avoid duplicating the config into other summary files to reduce divergence risk.

Usage:

After running `tools/build_dataset_v4.sh`, inspect `datasets/<dataset_id>/build_info.json` to verify the recorded git commit and SHA256 checksums for the config and splits.
