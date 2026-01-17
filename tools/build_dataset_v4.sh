#!/usr/bin/env bash
set -euo pipefail

# Build a v4 dataset using augment_dataset_v4.py and make_splits.py
# Usage: tools/build_dataset_v4.sh --dataset_id NAME --config base_config.json --target-size 30000 --out_root datasets --fast 0

DATASET_ID=${1:-dads_v4_30k_4s}
# BASE_CONFIG: if not passed, resolve from config.py (CONFIG_DATASET_PATH) or fall back
# to repository default. We fill BASE_CONFIG later after resolving DEFAULT_CFG.
BASE_CONFIG_PLACEHOLDER=1
# pool args may be passed positionally; prefer explicit env overrides; otherwise resolve
# from `config.py` defaults so builds are portable across machines.
NOISE_ARG=${3:-""}
RIR_ARG=${4:-""}
HARDNEG_ARG=${5:-""}
TARGET_SIZE=${6:-30000}
FAST=${7:-0}  # if non-zero, generate small sample for verification

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Resolve canonical V4 output root and default config from config.py when available.
# Falls back to repo-root `datasets` and the repo augment_config_v4.json if config.py isn't importable.
V4_OUT_ROOT="$(python3 - <<PY
try:
  import config
  p = getattr(config, 'V4_DATASETS_DIR', None)
  if p is None:
    from pathlib import Path
    print((Path(__file__).resolve().parents[1] / 'datasets').as_posix())
  else:
    print(str(p))
except Exception:
  from pathlib import Path
  print((Path(__file__).resolve().parents[1] / 'datasets').as_posix())
PY
)"

DEFAULT_CFG="$(python3 - <<PY
try:
  import config
  print(str(getattr(config, 'CONFIG_DATASET_PATH')))
except Exception:
  from pathlib import Path
  print((Path(__file__).resolve().parents[1] / '0 - DADS dataset extraction' / 'augment_config_v4.json').as_posix())
PY
)"

# If the caller provided an override config use it; otherwise use the project default
BASE_CONFIG=${2:-"${DEFAULT_CFG}"}

# OUT_ROOT: where v4 datasets are written (default to V4_OUT_ROOT resolved above)
OUT_ROOT=${OUT_ROOT:-"${V4_OUT_ROOT}"}

# Resolve pools: prefer positional args, then environment variables, then
# fall back to values declared in `config.py`, and finally repository defaults.
# NOISE_ARG / RIR_ARG / HARDNEG_ARG are positional args (may be empty).
if [ -n "$NOISE_ARG" ]; then
  NOISE_DIR="$NOISE_ARG"
elif [ -n "${NOISE_POOL_DIR:-}" ]; then
  NOISE_DIR="$NOISE_POOL_DIR"
else
  cfg_noise=$(python3 - <<PY
try:
  import config
  print(str(getattr(config, 'NOISE_POOL_DIR', '')))
except Exception:
  print('')
PY
)
  NOISE_DIR=${cfg_noise:-"0 - DADS dataset extraction/pools/noise"}
fi

if [ -n "$RIR_ARG" ]; then
  RIR_DIR="$RIR_ARG"
elif [ -n "${RIR_POOL_DIR:-}" ]; then
  RIR_DIR="$RIR_POOL_DIR"
else
  cfg_rir=$(python3 - <<PY
try:
  import config
  print(str(getattr(config, 'RIR_POOL_DIR', '')))
except Exception:
  print('')
PY
)
  RIR_DIR=${cfg_rir:-"augment/v4/rirs"}
fi

if [ -n "$HARDNEG_ARG" ]; then
  HARDNEG_DIR="$HARDNEG_ARG"
elif [ -n "${HARDNEG_POOL_DIR:-}" ]; then
  HARDNEG_DIR="$HARDNEG_POOL_DIR"
else
  cfg_hardneg=$(python3 - <<PY
try:
  import config
  print(str(getattr(config, 'HARDNEG_POOL_DIR', '')))
except Exception:
  print('')
PY
)
  HARDNEG_DIR=${cfg_hardneg:-"0 - DADS dataset extraction/pools/hardneg"}
fi

echo "Using pools: NOISE_DIR=${NOISE_DIR}, RIR_DIR=${RIR_DIR}, HARDNEG_DIR=${HARDNEG_DIR}"

OUT="${OUT_ROOT}/${DATASET_ID}"

# Safety: ensure clean output when requested. If OUT exists and OVERWRITE!=1, fail fast.
if [ -d "${OUT}" ]; then
  if [ "${OVERWRITE:-0}" != "1" ]; then
    echo "ERROR: Output exists: ${OUT}. Set OVERWRITE=1 to overwrite." >&2
    exit 2
  fi
  echo "[OVERWRITE=1] Removing entire dataset dir: ${OUT}"
  rm -rf "${OUT}"
fi

mkdir -p "${OUT}"

# write build_config.json into the OUT after ensuring OUT is clean
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

# Split counts: derive deterministic split counts and embed for auditability
target = int(${TARGET_SIZE})
rat = cfg.get('DEFAULT_SPLIT_RATIOS', {'train':0.8, 'val':0.1, 'test':0.1})
try:
    test_ratio = float(rat.get('test', 0.1))
    val_ratio = float(rat.get('val', 0.1))
    train_ratio = float(rat.get('train', 0.8))
except Exception:
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

n_test = max(1, int(round(target * test_ratio)))
n_val = max(1, int(round(target * val_ratio)))
n_train = target - n_val - n_test
if n_train < 1:
    n_train = 1
    rem = target - n_train
    if rem <= 0:
        n_val, n_test = 0, 0
    else:
        # keep test>=1 if possible
        n_test = min(max(1, n_test), rem)
        n_val = rem - n_test
if n_train + n_val + n_test != target:
    n_test = target - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train + n_val + n_test > target:
            n_val = max(0, target - n_train - n_test)

# write override config
cfg['output'] = cfg.get('output',{})
cfg['output']['samples_per_category_drone'] = per_cat
cfg['output']['samples_per_category_no_drone'] = per_cat
cfg['advanced'] = cfg.get('advanced',{})
cfg['advanced']['random_seed'] = int(cfg.get('advanced', {}).get('random_seed', 999))
# set audio duration
cfg['audio_parameters'] = cfg.get('audio_parameters',{})
cfg['audio_parameters']['duration_s'] = float(cfg.get('audio_parameters', {}).get('duration_s', 4.0))

# embed target + split ratios/counts for auditability
cfg['output']['dataset_target_size'] = int(target)
cfg['output']['split_ratios'] = {'train': float(train_ratio), 'val': float(val_ratio), 'test': float(test_ratio)}
cfg['output']['split_counts'] = {'train': int(n_train), 'val': int(n_val), 'test': int(n_test)}

# write temp config
out_root = Path(r"${OUT_ROOT}") / "${DATASET_ID}"
out_root.mkdir(parents=True, exist_ok=True)
cfg_path = out_root / 'build_config.json'
cfg_path.write_text(json.dumps(cfg), encoding='utf-8')
print('Wrote build config to', cfg_path)
print('Per-category samples:', per_cat)
print('Embedded split_counts:', cfg['output']['split_counts'])
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
