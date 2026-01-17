#!/usr/bin/env python3
"""Split augmented samples into dataset_train/dataset_val/dataset_test deterministically.

Usage: python3 tools/split_dataset.py --dataset <path> [--config <cfg.json>] [--apply]

This tool performs a dry-run by default. Pass --apply to move files and write per-split JSONL.
"""
from pathlib import Path
import argparse
import json
import shutil
import os
import sys
import math
import numpy as np


def load_config(cfg_path: Path):
    if not cfg_path or not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def collect_entries(dataset_root: Path):
    # find any augmentation_samples.jsonl under dataset_root (recursively)
    entries = []
    jsonl_paths = list(dataset_root.rglob('augmentation_samples.jsonl'))
    for p in sorted(jsonl_paths):
        for line in p.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # normalize shape
            train_meta = obj.get('train_meta') or obj
            entries.append({'train_meta': train_meta, 'debug_meta': obj.get('debug_meta'), 'src_jsonl': p})
    return entries


def deterministic_shuffle(entries, master_seed: int):
    rng = np.random.default_rng(int(master_seed or 0))
    idx = rng.permutation(len(entries))
    return [entries[i] for i in idx]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def move_audio_and_emit(dataset_root: Path, entry, dst_split: str, apply: bool):
    train_meta = entry['train_meta']
    rel = train_meta.get('relpath')
    if not rel:
        return False, 'no relpath'
    src = (dataset_root / rel).resolve()
    # compute filename and label
    parts = rel.split('/')
    filename = parts[-1]
    label = str(train_meta.get('label', 0))
    dst_dir = dataset_root / dst_split / label
    ensure_dir(dst_dir)
    dst = dst_dir / filename
    if apply:
        # ensure source exists
        if not src.exists():
            return False, f'src-missing:{src}'
        # use atomic move
        try:
            os.replace(str(src), str(dst))
        except Exception as e:
            try:
                shutil.move(str(src), str(dst))
            except Exception as e2:
                return False, f'move-failed:{e}|{e2}'
    return True, str(dst)


def write_jsonl_line(path: Path, obj, apply: bool):
    if not apply:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + '\n')
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--config', default=None)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()

    ds = Path(args.dataset)
    if not ds.exists():
        print('Dataset not found:', ds)
        return 2

    cfg = load_config(Path(args.config) if args.config else (ds / 'build_config.json'))
    master_seed = int(cfg.get('advanced', {}).get('random_seed', 0)) if cfg else int(0)

    entries = collect_entries(ds)
    total = len(entries)
    if total == 0:
        print('No entries found under', ds)
        return 1

    # compute split counts
    split_ratio = cfg.get('validation_strategy', {}).get('split_ratio', 0.8) if cfg else 0.8
    train_n = int(math.floor(total * float(split_ratio)))
    remainder = total - train_n
    val_n = int(math.floor(remainder / 2.0))
    test_n = remainder - val_n
    # ensure minimal 1 each when possible
    if train_n <= 0 and total >= 1:
        train_n = max(1, total - 2)
    if val_n == 0 and total >= 3:
        val_n = 1
    if test_n == 0 and total >= 3:
        test_n = 1

    print(f'Total entries: {total} -> train={train_n}, val={val_n}, test={test_n}')

    entries_shuffled = deterministic_shuffle(entries, master_seed)

    assignments = {}
    i = 0
    for e in entries_shuffled:
        if i < train_n:
            assignments[id(e)] = 'dataset_train'
        elif i < train_n + val_n:
            assignments[id(e)] = 'dataset_val'
        else:
            assignments[id(e)] = 'dataset_test'
        i += 1

    # prepare per-split jsonl paths
    jsonl_paths = {
        'dataset_train': ds / 'dataset_train' / 'augmentation_samples.jsonl',
        'dataset_val': ds / 'dataset_val' / 'augmentation_samples.jsonl',
        'dataset_test': ds / 'dataset_test' / 'augmentation_samples.jsonl'
    }

    # dry-run: list moves
    actions = []
    for e in entries_shuffled:
        dst = assignments[id(e)]
        train_meta = e['train_meta']
        ok, info = move_audio_and_emit(ds, e, dst, apply=False)
        actions.append((train_meta.get('relpath'), dst, ok, info))

    print('\nPlanned actions:')
    for a in actions:
        print('  ', a[0], '->', a[1], 'ok' if a[2] else 'ERR', a[3])

    if not args.apply:
        print('\nDry-run complete. Re-run with --apply to perform moves and write JSONL')
        return 0

    # apply: move files and write per-split jsonl
    # backup original jsonl files
    for pth in set([e['src_jsonl'] for e in entries]):
        bak = pth.with_suffix(pth.suffix + '.bak')
        if not bak.exists():
            shutil.copy2(str(pth), str(bak))

    # process and write
    for e in entries_shuffled:
        dst = assignments[id(e)]
        train_meta = e['train_meta']
        debug_meta = e.get('debug_meta')
        ok, info = move_audio_and_emit(ds, e, dst, apply=True)
        if not ok:
            print('MOVE FAILED for', train_meta.get('relpath'), info)
            continue
        # update relpath to new location relative to ds
        parts = train_meta.get('relpath').split('/')
        filename = parts[-1]
        label = str(train_meta.get('label', 0))
        new_rel = f"{dst}/{label}/{filename}"
        train_meta['relpath'] = new_rel
        out_obj = {'train_meta': train_meta}
        if debug_meta is not None:
            out_obj['debug_meta'] = debug_meta
        write_jsonl_line(jsonl_paths[dst], out_obj, apply=True)

    # write splits.json
    splits = {'train': train_n, 'val': val_n, 'test': test_n}
    (ds / 'splits.json').write_text(json.dumps(splits, indent=2), encoding='utf-8')
    print('Applied split; wrote splits.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
