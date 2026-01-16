#!/usr/bin/env python3
"""Create a deterministic split assignment for drone sources.

Produces a `splits.json` mapping source_id -> split_name under datasets/<dataset_id>/splits.json
Usage: tools/make_splits.py --dataset_id myds --out_root datasets --seed 42 --train 0.8 --val 0.1 --test 0.1
"""
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_id', required=True)
    p.add_argument('--out_root', default='datasets')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train', type=float, default=0.8)
    p.add_argument('--val', type=float, default=0.1)
    p.add_argument('--test', type=float, default=0.1)
    args = p.parse_args()

    out_root = Path(args.out_root)
    ds_dir = out_root / args.dataset_id
    ds_dir.mkdir(parents=True, exist_ok=True)
    splits_path = ds_dir / 'splits.json'

    # discover drone source ids from offline folder if present
    try:
        import config as project_config
        offline_dir = Path(project_config.DATASET_DADS_OFFLINE_DIR)
        drone_dir = offline_dir / '1'
        sources = sorted([p.stem for p in drone_dir.glob('*.wav')]) if drone_dir.exists() else []
    except Exception:
        sources = []

    if not sources:
        # produce empty mapping
        splits_path.write_text(json.dumps({}), encoding='utf-8')
        print('No sources found; wrote empty splits.json at', splits_path)
        return 0

    rng = np.random.default_rng(int(args.seed))
    n = len(sources)
    # shuffle deterministically
    idx = rng.permutation(n)
    train_cut = int(args.train * n)
    val_cut = train_cut + int(args.val * n)
    mapping = {}
    for pos, i in enumerate(idx):
        src = sources[i]
        if pos < train_cut:
            mapping[src] = 'train'
        elif pos < val_cut:
            mapping[src] = 'val'
        else:
            mapping[src] = 'test'

    splits_path.write_text(json.dumps(mapping, indent=2), encoding='utf-8')
    print('Wrote splits.json with', len(mapping), 'entries to', splits_path)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
