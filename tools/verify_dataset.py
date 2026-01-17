#!/usr/bin/env python3
"""Sanity checks and provenance verification for a v4 dataset.

Usage: python3 tools/verify_dataset.py --dataset <path>
"""
from pathlib import Path
import argparse
import json
import hashlib
import os


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as fh:
        while True:
            b = fh.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def count_wavs(split_dir: Path):
    c0 = len(list((split_dir / '0').glob('*.wav'))) if (split_dir / '0').exists() else 0
    c1 = len(list((split_dir / '1').glob('*.wav'))) if (split_dir / '1').exists() else 0
    return c0, c1


def count_jsonl(path: Path):
    if not path.exists():
        return 0
    return sum(1 for _ in path.open('r', encoding='utf-8') if _.strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    args = p.parse_args()

    ds = Path(args.dataset)
    if not ds.exists():
        print('Dataset not found:', ds)
        return 2

    print('Verifying dataset:', ds)
    # provenance files
    for fname in ('build_config.json', 'effective_config.json', 'build_info.json', 'splits.json'):
        fp = ds / fname
        if fp.exists():
            print(f'  {fname}: present, sha256={sha256(fp)}')
        else:
            print(f'  {fname}: MISSING')

    ok = True
    for split in ('dataset_train', 'dataset_val', 'dataset_test'):
        sd = ds / split
        jl = sd / 'augmentation_samples.jsonl'
        jcount = count_jsonl(jl)
        c0, c1 = count_wavs(sd)
        print(f'  Split {split}: jsonl_lines={jcount}, wavs class0={c0}, class1={c1}')
        if jcount != (c0 + c1):
            print('    WARNING: jsonl count != file count')
            ok = False
        if c0 == 0 or c1 == 0:
            print('    WARNING: class 0 or 1 empty')
            ok = False

    if ok:
        print('\nVerification PASSED')
        return 0
    else:
        print('\nVerification FAILED (see warnings)')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
