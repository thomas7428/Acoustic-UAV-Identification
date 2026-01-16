#!/usr/bin/env bash
set -euo pipefail

# Build a v4 dataset using augment_dataset_v4.py and make_splits.py
# Usage: tools/build_dataset_v4.sh --dataset_id NAME --config base_config.json --target-size 30000 --out_root datasets --fast 0

DATASET_ID=${1:-dads_v4_30k_4s}
BASE_CONFIG=${2:-0 - DADS dataset extraction/augment_config_v4.json}
NOISE_DIR=${3:-"0 - DADS dataset extraction/noise_pool"}
RIR_DIR=${4:-"augment/v4/rirs"}
HARDNEG_DIR=${5:-"0 - DADS dataset extraction/hard_negatives"}
TARGET_SIZE=${6:-30000}
FAST=${7:-0}  # if non-zero, generate small sample for verification

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# compute a small target if FAST
if [ "$FAST" != "0" ]; then
  TARGET_SIZE=200
  echo "FAST mode: using TARGET_SIZE=${TARGET_SIZE} for quick verification"
fi

# ensure splits.json exists
python3 - <<PYCODE
import json,sys
from pathlib import Path
base = Path(r"${BASE_CONFIG}")
if not base.exists():
    print('Base config not found:', base)
    sys.exit(2)
cfg = json.loads(base.read_text(encoding='utf-8'))
# compute per-category counts
n_drone = len(cfg.get('drone_augmentation',{}).get('categories',[]))
n_no = len(cfg.get('no_drone_augmentation',{}).get('categories',[]))
per_cat = 1
if n_drone + n_no > 0:
    per_cat = max(1, int(int(${TARGET_SIZE}) // max(1, (n_drone + n_no))))
# write override config
cfg['output'] = cfg.get('output',{})
cfg['output']['samples_per_category_drone'] = per_cat
cfg['output']['samples_per_category_no_drone'] = per_cat
cfg['advanced'] = cfg.get('advanced',{})
cfg['advanced']['random_seed'] = int(cfg.get('advanced', {}).get('random_seed', 999))
# set audio duration
cfg['audio_parameters'] = cfg.get('audio_parameters',{})
cfg['audio_parameters']['duration_s'] = float(cfg.get('audio_parameters', {}).get('duration_s', 4.0))
# write temp config
out_root = Path(r"${OUT_ROOT}") / "${DATASET_ID}"
out_root.mkdir(parents=True, exist_ok=True)
cfg_path = out_root / 'build_config.json'
cfg_path.write_text(json.dumps(cfg), encoding='utf-8')
print('Wrote build config to', cfg_path)
print('Per-category samples:', per_cat)
PYCODE

# generate splits (use provided pools)
python3 -u tools/make_splits.py --dataset_id "${DATASET_ID}" --out_root "${OUT_ROOT}" --noise_dir "${NOISE_DIR}" --rir_dir "${RIR_DIR}" --hardneg_dir "${HARDNEG_DIR}" --ood_noise_frac 0.15 --ood_rir_frac 0.15 --hardneg_frac 0.10

# Persist effective config and build metadata for reproducibility
OUT="${OUT_ROOT}/${DATASET_ID}"
mkdir -p "${OUT}"
# cfg used by generator
CFG_USED="${OUT}/build_config.json"
if [ -f "${CFG_USED}" ]; then
  cp "${CFG_USED}" "${OUT}/effective_config.json"

  GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  TS_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  CFG_SHA="$(python3 - <<PY
import hashlib
print(hashlib.sha256(open('${OUT}/effective_config.json','rb').read()).hexdigest())
PY
)"
  SPLITS_SHA="$(python3 - <<PY
import hashlib
print(hashlib.sha256(open('${OUT}/splits.json','rb').read()).hexdigest())
PY
)"

  cat > "${OUT}/build_info.json" <<JSON
{
  "dataset_id": "${DATASET_ID}",
  "git_sha": "${GIT_SHA}",
  "timestamp_utc": "${TS_UTC}",
  "command": "$(printf "%q " "$0" "$@")",
  "env": {
    "PYTHONHASHSEED": "${PYTHONHASHSEED:-}",
    "OMP_NUM_THREADS": "${OMP_NUM_THREADS:-}",
    "MKL_NUM_THREADS": "${MKL_NUM_THREADS:-}",
    "OPENBLAS_NUM_THREADS": "${OPENBLAS_NUM_THREADS:-}",
    "NUMEXPR_NUM_THREADS": "${NUMEXPR_NUM_THREADS:-}",
    "FAST": "${FAST:-}"
  },
  "sha256": {
    "effective_config.json": "${CFG_SHA}",
    "splits.json": "${SPLITS_SHA}"
  }
}
JSON
  echo "Wrote effective_config.json and build_info.json to ${OUT}"
else
  echo "Warning: config ${CFG_USED} not found; skipping effective_config/build_info write"
fi

# run generator
PYTHONPATH=. python3 -u augment_dataset_v4.py --config "${OUT_ROOT}/${DATASET_ID}/build_config.json" --out_dir "${OUT_ROOT}/${DATASET_ID}"

# run strict sanity on produced dataset
PYTHONPATH=. python3 -u tools/run_v4_dataset_sanity.py --strict

echo "Build complete for ${DATASET_ID}"
exit 0
