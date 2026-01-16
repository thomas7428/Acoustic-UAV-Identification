#!/usr/bin/env bash
set -euo pipefail
# Minimal CI runner for v4 dataset sanity (strict mode)
PYTHONPATH=. python3 tools/run_v4_dataset_sanity.py --strict --max_leak_auc 0.55 --max_clip_count 0 --min_unique_rir 3 --min_unique_scene 3 --max_stem_dominance 0.25
echo "CI v4 sanity passed"
