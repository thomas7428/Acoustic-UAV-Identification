import json
import argparse
from pathlib import Path
from typing import List

from termcolor import colored

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from tools import perf_utils
import numpy as np


# Saving results into a json file using perf_utils
def save_predictions(split: str, json_path: str, limit: int = None):
    """Generate predictions for files in `split` (train/val/test) and save to `json_path`.
    If `limit` is provided, it limits the number of files evaluated (split evenly across classes).
    """
    split = split.lower()
    if split == 'test':
        dataset_dir = config.DATASET_TEST_DIR
    elif split == 'val':
        dataset_dir = config.DATASET_VAL_DIR
    elif split == 'train':
        dataset_dir = config.DATASET_TRAIN_DIR
    else:
        raise ValueError(f"Unknown split: {split}")

    class0_dir = Path(dataset_dir) / "0"
    class1_dir = Path(dataset_dir) / "1"
    files0 = sorted([p for p in class0_dir.glob("*.wav")]) if class0_dir.exists() else []
    files1 = sorted([p for p in class1_dir.glob("*.wav")]) if class1_dir.exists() else []

    # Determine per-class limit
    if limit is not None and limit > 0:
        per_class = max(1, limit // 2)
        files0 = files0[:per_class]
        files1 = files1[:per_class]

    filenames = [p.name for p in files0] + [p.name for p in files1]
    labels = [0] * len(files0) + [1] * len(files1)

    # Strict load: use the full-data file only (mel_{split}.json) and require
    # an exact mapping from filename -> mel. No fallbacks, no index files.
    # This enforces the user's requirement: testers must not modify or
    # heuristically map precomputed features at runtime.
    data_map = {}
    # Determine data path for the split
    if split == 'test':
        data_path = Path(config.MEL_TEST_DATA_PATH)
    elif split == 'val':
        data_path = Path(config.MEL_VAL_DATA_PATH)
    elif split == 'train':
        data_path = Path(config.MEL_TRAIN_DATA_PATH)
    else:
        raise ValueError(f"Unknown split: {split}")

    if not data_path.exists():
        raise RuntimeError(f"Required full-data features file not found: {data_path}")

    # Load and expect the full-data container: {'mel': [...], 'mapping': [...]} or similar
    with open(data_path, 'r') as fp:
        j = json.load(fp)

    if not (isinstance(j, dict) and 'mel' in j and isinstance(j['mel'], list)):
        raise RuntimeError(f"Full-data features file {data_path} does not contain a 'mel' list as expected")

    mels = j['mel']
    # Try to find a names list in common keys (strict: must exist and match length)
    candidate_name_keys = ['mapping', 'names', 'file_names', 'files']
    names = None
    for nk in candidate_name_keys:
        if nk in j and isinstance(j[nk], list) and len(j[nk]) == len(mels):
            names = j[nk]
            break
    if names is None:
        raise RuntimeError(f"Full-data features file {data_path} missing a names mapping; cannot proceed in strict mode")

    data_map = {n: np.array(m, dtype=np.float32).squeeze() for n, m in zip(names, mels)}

    if config.PRECOMPUTED_ONLY and not data_map:
        raise RuntimeError(f"PRECOMPUTED_ONLY is True but full-data mapping is empty for split={split}")

    # Build the requested model
    model_key = 'CNN'
    # The CLI may set a different model; we'll set that outside this helper
    model = perf_utils.build_model_from_config(model_key)

    # Batch predict sequentially (small batches to limit memory)
    all_paths = [*files0, *files1]
    batch_size = 32
    scores: List[float] = []
    names_out: List[str] = []

    # Use the strict data_map loaded above. Require exact filename matches.
    all_filenames = [p.name for p in all_paths]
    missing_keys = [fn for fn in all_filenames if fn not in data_map]
    if missing_keys:
        raise RuntimeError(
            f"Missing precomputed MELs for {len(missing_keys)} files in split={split}: {missing_keys[:5]}... \n"
            "No fallback is performed in strict mode. Ensure the full-data features contain exact filenames."
        )

    for i in range(0, len(all_paths), batch_size):
        batch_paths = all_paths[i:i+batch_size]
        X_batch = []
        batch_names = []
        for p in batch_paths:
            name = p.name
            mel = data_map.get(name)
            if mel is None:
                raise RuntimeError(f"Missing precomputed MEL for '{name}' in split={split}; strict mode prevents fallback.")
            # Use the precomputed mel as-is (do not pad/transpose/modify)
            X_batch.append(np.array(mel, dtype=np.float32))
            batch_names.append(name)
        if not X_batch:
            continue
        X_batch = np.stack(X_batch, axis=0)[..., np.newaxis]
        # Validate model input shape vs supplied batch shape (excluding batch dim)
        model_input_shape = tuple(s for s in getattr(model, 'input_shape', (None,))[1:])
        supplied_shape = tuple(X_batch.shape[1:])
        if model_input_shape and model_input_shape != supplied_shape:
            raise RuntimeError(
                f"Model expected input shape {model_input_shape} but received {supplied_shape} from precomputed MELs."
            )

        probs = perf_utils.predict_batch(model, X_batch, batch_size=batch_size)
        for nm, pr in zip(batch_names, probs.tolist()):
            names_out.append(nm)
            scores.append(float(pr))

    # Convert scores to binary predictions using thresholds
    threshold = config.MODEL_THRESHOLDS.get('CNN', 0.5)
    preds_bin = [1 if s > threshold else 0 for s in scores]

    perf_utils.save_predictions_json(Path(json_path), names_out, scores, threshold=threshold)

    return labels, preds_bin, scores, names_out


# Calculating performance scores using perf_utils
def performance_calcs(performance_path: str, y_true, y_pred):
    # Delegate to perf_utils for consistent formatting
    from pathlib import Path
    perf = perf_utils.calc_and_save_perf_scores(Path(performance_path), y_true, y_pred)
    return perf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CNN on dataset split')
    parser.add_argument('--limit', type=int, default=0, help='Limit total number of files to evaluate (split across classes)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Dataset split to evaluate')
    parser.add_argument('--output', type=str, default=config.CNN_PREDICTIONS_PATH_STR, help='Output JSON path for predictions')
    args = parser.parse_args()

    limit = args.limit if args.limit and args.limit > 0 else None
    labels, preds_bin, scores, filenames = save_predictions(args.split, args.output, limit=limit)
    perf = performance_calcs(config.CNN_SCORES_PATH_STR, labels, preds_bin)

    print(colored(f"CNN model performance scores have been saved to {config.CNN_SCORES_PATH_STR}.", "green"))
    print(json.dumps(perf, indent=2))
