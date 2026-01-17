#!/usr/bin/env python3
"""Create a deterministic split assignment for drone sources.

Produces a `splits.json` mapping source_id -> split_name under datasets/<dataset_id>/splits.json
Usage: tools/make_splits.py --dataset_id myds --out_root datasets --seed 42 --train 0.8 --val 0.1 --test 0.1
"""
import argparse
import json
from pathlib import Path
import numpy as np
import sys

# Ensure repo root is importable when script is invoked from tools/ (so `import config` works)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_id', required=True)
    p.add_argument('--out_root', default='datasets')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train', type=float, default=0.8)
    p.add_argument('--val', type=float, default=0.1)
    p.add_argument('--test', type=float, default=0.1)
    p.add_argument('--noise_dir', default=None, help='Path to noise pool for OOD noise selection')
    p.add_argument('--rir_dir', default=None, help='Path to RIR bank for OOD RIR selection')
    p.add_argument('--hardneg_dir', default=None, help='Path to hard-negative pool')
    p.add_argument('--ood_noise_frac', type=float, default=0.15, help='Fraction of noise files reserved for test_ood_noise')
    p.add_argument('--ood_rir_frac', type=float, default=0.15, help='Fraction of RIRs reserved for test_ood_rir')
    p.add_argument('--hardneg_frac', type=float, default=0.10, help='Fraction of hardnegatives reserved for test_hardneg')
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
    except Exception as e:
        print('make_splits: failed to discover sources:', e)
        sources = []

    if not sources:
        # still attempt to create OOD lists if directories provided
        mapping = {}
    else:
        mapping = {}
        rng = np.random.default_rng(int(args.seed))
        n = len(sources)
        # shuffle deterministically
        idx = rng.permutation(n)

        # Compute desired counts per split robustly and ensure at least one sample per split when possible
        # Start with rounded counts
        n_test = max(1, int(round(args.test * n)))
        n_val = max(1, int(round(args.val * n)))
        n_train = n - n_val - n_test
        # If rounding produced non-positive train, fix by reducing val/test
        if n_train < 1:
            # Ensure train at least 1
            n_train = 1
            rem = n - n_train
            # allocate remaining to val and test proportionally to their ratios
            total_ratio = args.val + args.test
            if total_ratio <= 0:
                n_val = 0
                n_test = rem
            else:
                n_val = max(1, int(round((args.val / total_ratio) * rem))) if rem > 1 else 0
                n_test = rem - n_val
        # final safety: ensure sum equals n (adjust test)
        if n_train + n_val + n_test != n:
            n_test = n - n_train - n_val
            if n_test < 1:
                n_test = 1
                # adjust val down if needed
                if n_train + n_val + n_test > n:
                    n_val = max(0, n - n_train - n_test)

        train_cut = n_train
        val_cut = n_train + n_val

        for pos, i in enumerate(idx):
            src = sources[i]
            if pos < train_cut:
                mapping[src] = 'train'
            elif pos < val_cut:
                mapping[src] = 'val'
            else:
                mapping[src] = 'test'

    # OOD selections
    ood_noise = []
    ood_rir = []
    test_hardneg = []
    rng = np.random.default_rng(int(args.seed))
    # noise pool
    if args.noise_dir:
        noise_p = Path(args.noise_dir)
        if noise_p.exists():
            noise_files = sorted([p.name for p in noise_p.glob('*') if p.is_file()])
            k = max(1, int(len(noise_files) * float(args.ood_noise_frac)))
            sel_idx = rng.choice(len(noise_files), size=k, replace=False)
            ood_noise = [noise_files[i] for i in sel_idx]
    # rir pool
    if args.rir_dir:
        rir_p = Path(args.rir_dir)
        if rir_p.exists():
            rir_files = sorted([p.name for p in rir_p.glob('*.wav')])
            k = max(1, int(len(rir_files) * float(args.ood_rir_frac))) if rir_files else 0
            if k > 0:
                sel_idx = rng.choice(len(rir_files), size=k, replace=False)
                ood_rir = [rir_files[i] for i in sel_idx]
    # hard negatives
    if args.hardneg_dir:
        hn_p = Path(args.hardneg_dir)
        if hn_p.exists():
            hn_files = sorted([p.name for p in hn_p.glob('*') if p.is_file()])
            k = max(1, int(len(hn_files) * float(args.hardneg_frac))) if hn_files else 0
            if k > 0:
                sel_idx = rng.choice(len(hn_files), size=k, replace=False)
                test_hardneg = [hn_files[i] for i in sel_idx]

    # embed OOD lists into mapping under reserved keys
    mapping['_ood_noise'] = ood_noise
    mapping['_ood_rir'] = ood_rir
    mapping['_test_hardneg'] = test_hardneg

    splits_path.write_text(json.dumps(mapping, indent=2), encoding='utf-8')
    print('Wrote splits.json with', len([k for k in mapping.keys() if not k.startswith('_')]), 'source entries to', splits_path)
    print('OOD noise count:', len(ood_noise), 'OOD rir count:', len(ood_rir), 'test_hardneg count:', len(test_hardneg))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
