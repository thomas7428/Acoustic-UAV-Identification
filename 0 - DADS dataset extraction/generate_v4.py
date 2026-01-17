#!/usr/bin/env python3
"""Minimal v4 dataset generator wrapper placed in the extraction root.

Usage:
  python3 "0 - DADS dataset extraction/generate_v4.py" <dataset_id> [--target SIZE] [--fast]

Behavior:
 - Resolves pool dirs from env vars or `config.py` defaults.
 - Writes dataset under `config.EXTRACTION_DIR / <dataset_id>` (directly in the root).
 - Creates a minimal `build_config.json`, runs `tools/make_splits.py`, then
   calls `augment_dataset_v4.py` to generate audio and per-split JSONL.
 - Writes `effective_config.json` and `build_info.json` for provenance.

This wrapper purposely keeps logic tiny so the pipeline is easy to inspect.
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import config


def resolve_pool(env_name, cfg_attr, default_subpath):
    val = os.environ.get(env_name)
    if val:
        return Path(val)
    return getattr(config, cfg_attr, config.EXTRACTION_DIR / default_subpath)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset_id')
    p.add_argument('--target', type=int, default=int(os.environ.get('DATASET_TARGET_SIZE', 2000)))
    p.add_argument('--fast', action='store_true')
    args = p.parse_args()

    out_root = config.EXTRACTION_DIR
    dataset_id = args.dataset_id
    out = out_root / dataset_id
    if out.exists():
        print('Overwriting existing dataset at', out)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # resolve pools
    noise = resolve_pool('NOISE_POOL_DIR', 'NOISE_POOL_DIR', 'pools/noise')
    rirs = resolve_pool('RIR_POOL_DIR', 'RIR_POOL_DIR', 'pools/rirs')
    hardneg = resolve_pool('HARDNEG_POOL_DIR', 'HARDNEG_POOL_DIR', 'pools/hardneg')

    print('Using pools:')
    print('  NOISE:', noise)
    print('  RIRS: ', rirs)
    print('  HARDNEG:', hardneg)

    # prepare a minimal build_config by copying project default and embedding target
    base_cfg = Path(config.CONFIG_DATASET_PATH)
    if not base_cfg.exists():
        raise SystemExit('Base config not found: ' + str(base_cfg))
    cfg = json.loads(base_cfg.read_text(encoding='utf-8'))
    cfg.setdefault('output', {})
    cfg['output']['dataset_target_size'] = int(args.target)
    cfg['output']['split_counts'] = cfg['output'].get('split_counts', {})
    # write build_config.json
    bc_path = out / 'build_config.json'
    bc_path.write_text(json.dumps(cfg), encoding='utf-8')
    print('Wrote', bc_path)

    # run make_splits.py
    cmd = [sys.executable, '-u', 'tools/make_splits.py', '--dataset_id', dataset_id, '--out_root', str(out_root), '--noise_dir', str(noise), '--rir_dir', str(rirs), '--hardneg_dir', str(hardneg), '--ood_noise_frac', '0.15', '--ood_rir_frac', '0.15', '--hardneg_frac', '0.10']
    if args.fast:
        cmd += ['--fast']
    subprocess.check_call(cmd)

    # persist effective_config.json and build_info.json (simple form)
    shutil.copy2(str(bc_path), str(out / 'effective_config.json'))
    with open(out / 'build_info.json', 'w', encoding='utf-8') as fh:
        json.dump({'dataset_id': dataset_id, 'command': 'generate_v4.py', 'pools': {'noise': str(noise), 'rirs': str(rirs), 'hardneg': str(hardneg)}}, fh, indent=2)

    # run augment_dataset_v4.py
    cmd2 = [sys.executable, '-u', 'augment_dataset_v4.py', '--config', str(bc_path), '--out_dir', str(out)]
    if args.fast:
        cmd2 += ['--fast']
    subprocess.check_call(cmd2)

    print('Generation complete:', out)


if __name__ == '__main__':
    main()
