"""
Dataset organization helper.
Creates and manages dataset folders under the extraction directory:
- dataset_dads       : original downloaded files (preserve originals)
- dataset_augment    : augmented files (before distribution)
- dataset_train      : training split (class subfolders 0/1)
- dataset_val        : validation split
- dataset_test       : test split (held-out)

Usage examples:
  python tools/organize_datasets.py init
  python tools/organize_datasets.py import-augment --src "0 - DADS dataset extraction/dataset_augmented" --move
  python tools/organize_datasets.py split --source "0 - DADS dataset extraction/dataset_augment" --val 0.1 --test 0.1 --dry-run

This script defaults to using `config.DATASET_ROOT.parent` (the extraction folder) as base.
"""
from pathlib import Path
import shutil
import argparse
import random
import os
import sys

# import project config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as project_config

EXTRACTION_DIR = Path(project_config.PROJECT_ROOT) / "0 - DADS dataset extraction"
DATASETS = {
    'dads': EXTRACTION_DIR / 'dataset_dads',
    'augment': EXTRACTION_DIR / 'dataset_augment',
    'train': EXTRACTION_DIR / 'dataset_train',
    'val': EXTRACTION_DIR / 'dataset_val',
    'test': EXTRACTION_DIR / 'dataset_test'
}
CLASS_SUBFOLDERS = ['0', '1']


def ensure_structure():
    for name, path in DATASETS.items():
        path.mkdir(parents=True, exist_ok=True)
        for c in CLASS_SUBFOLDERS:
            (path / c).mkdir(parents=True, exist_ok=True)
    print(f"Dataset folders ensured under: {EXTRACTION_DIR}")


def import_files(src: Path, dest_key: str, move=False, dry_run=False):
    """Copy or move all WAV files from src into DATASETS[dest_key], preserving class subfolders if present.
    If src contains '0' and '1', copy classwise; otherwise attempt to infer class from filename prefix.
    """
    dest = DATASETS[dest_key]
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    # if src has class subfolders
    if any((src / c).exists() for c in CLASS_SUBFOLDERS):
        for c in CLASS_SUBFOLDERS:
            s = src / c
            d = dest / c
            if not s.exists():
                continue
            files = list(s.glob('*.wav'))
            for f in files:
                target = d / f.name
                if dry_run:
                    print(f"DRY: {'mv' if move else 'cp'} {f} -> {target}")
                else:
                    if move:
                        shutil.move(str(f), str(target))
                    else:
                        shutil.copy2(str(f), str(target))
    else:
        # flat folder: try to infer class from filename (contains _drone_ or aug_ambient)
        files = list(src.glob('*.wav'))
        for f in files:
            # heuristic: if filename contains 'drone' or starts with '1_' go to class '1'
            lower = f.name.lower()
            if 'drone' in lower or lower.startswith('1_'):
                c = '1'
            else:
                c = '0'
            target = DATASETS[dest_key] / c / f.name
            if dry_run:
                print(f"DRY: {'mv' if move else 'cp'} {f} -> {target}")
            else:
                if move:
                    shutil.move(str(f), str(target))
                else:
                    shutil.copy2(str(f), str(target))

    print(f"Imported files from {src} into {dest} (move={move})")


def split_dataset(source_key: str, val_ratio: float, test_ratio: float, seed=42, move=False, dry_run=False):
    """Split files from DATASETS[source_key] into dataset_train/dataset_val/dataset_test.
    Ratios are fractions (e.g., val_ratio=0.1, test_ratio=0.1). Remaining goes to train.
    Splits are per-class to maintain balance.
    """
    random.seed(seed)
    src = DATASETS[source_key]
    if not src.exists():
        raise FileNotFoundError(f"Source dataset not found: {src}")

    for c in CLASS_SUBFOLDERS:
        files = list((src / c).glob('*.wav'))
        n = len(files)
        if n == 0:
            print(f"No files found for class {c} in {src}")
            continue
        random.shuffle(files)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test_files = files[:n_test]
        val_files = files[n_test:n_test + n_val]
        train_files = files[n_test + n_val:]

        def _do_move_list(lst, dest_key):
            dest_dir = DATASETS[dest_key] / c
            for f in lst:
                target = dest_dir / f.name
                if dry_run:
                    print(f"DRY: {'mv' if move else 'cp'} {f} -> {target}")
                else:
                    if move:
                        shutil.move(str(f), str(target))
                    else:
                        shutil.copy2(str(f), str(target))

        _do_move_list(test_files, 'test')
        _do_move_list(val_files, 'val')
        _do_move_list(train_files, 'train')

        print(f"Class {c}: total={n}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")


def status():
    for name, path in DATASETS.items():
        counts = {c: len(list((path / c).glob('*.wav'))) for c in CLASS_SUBFOLDERS}
        print(f"{name}: {counts}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Organize DADS dataset directories')
    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('init', help='Create dataset folders (dataset_dads, dataset_augment, train/val/test)')

    p_imp = sub.add_parser('import-augment', help='Import augmented files into dataset_augment')
    p_imp.add_argument('--src', required=True, help='Source augment folder')
    p_imp.add_argument('--move', action='store_true', help='Move files instead of copying')
    p_imp.add_argument('--dry-run', action='store_true')

    p_imp2 = sub.add_parser('import-dads', help='Import original downloads into dataset_dads')
    p_imp2.add_argument('--src', required=True, help='Source original files folder')
    p_imp2.add_argument('--move', action='store_true', help='Move files instead of copying')
    p_imp2.add_argument('--dry-run', action='store_true')

    p_split = sub.add_parser('split', help='Split a source dataset into train/val/test')
    p_split.add_argument('--source', required=True, choices=['augment','dads'], help='Which dataset to split from')
    p_split.add_argument('--val', type=float, default=0.1, help='Validation ratio')
    p_split.add_argument('--test', type=float, default=0.1, help='Test ratio')
    p_split.add_argument('--seed', type=int, default=42)
    p_split.add_argument('--move', action='store_true')
    p_split.add_argument('--dry-run', action='store_true')

    sub.add_parser('status', help='Show counts per dataset/class')

    args = parser.parse_args()

    if args.cmd == 'init':
        ensure_structure()
    elif args.cmd == 'import-augment':
        ensure_structure()
        import_files(Path(args.src), 'augment', move=args.move, dry_run=args.dry_run)
    elif args.cmd == 'import-dads':
        ensure_structure()
        import_files(Path(args.src), 'dads', move=args.move, dry_run=args.dry_run)
    elif args.cmd == 'split':
        ensure_structure()
        split_dataset(args.source, val_ratio=args.val, test_ratio=args.test, seed=args.seed, move=args.move, dry_run=args.dry_run)
    elif args.cmd == 'status':
        ensure_structure()
        status()
    else:
        parser.print_help()
