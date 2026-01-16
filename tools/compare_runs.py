#!/usr/bin/env python3
import json, hashlib
from pathlib import Path
import soundfile as sf
import numpy as np


def load_jsonl(p: Path):
    rows = []
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_pcm_wav(path: Path):
    """Hash decoded PCM (float32) of a WAV/PCM file.

    We prefer hashing decoded PCM because WAV container headers/chunks may
    differ run-to-run even when audio samples are identical. If decoding
    fails for any reason, fall back to hashing the raw file bytes.
    """
    try:
        x, sr = sf.read(path, dtype='float32', always_2d=False)
        x = np.asarray(x, dtype=np.float32)
        return hashlib.sha256(x.tobytes()).hexdigest()
    except Exception:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                h.update(chunk)
        return h.hexdigest()


def norm_meta(m: dict):
    drop = set(["generation_time", "timestamp", "duration_s", "notes"]) 
    return {k: v for k, v in m.items() if k not in drop}


def main(run1_dir, run2_dir, meta_name='augmentation_metadata.jsonl', max_print=30):
    run1_dir, run2_dir = Path(run1_dir), Path(run2_dir)
    # Autodetect meta file if not explicitly provided
    if meta_name is None:
        candidates = ['augmentation_samples.jsonl', 'augmentation_metadata.jsonl']
        for cand in candidates:
            if (run1_dir / cand).exists() and (run2_dir / cand).exists():
                meta_name = cand
                break
        else:
            raise FileNotFoundError(f"No metadata JSONL found in {run1_dir} or {run2_dir}")
    m1 = load_jsonl(run1_dir / meta_name)
    m2 = load_jsonl(run2_dir / meta_name)

    d1 = {r.get('relpath') or r.get('filename'): r for r in m1}
    d2 = {r.get('relpath') or r.get('filename'): r for r in m2}

    only1 = sorted(set(d1) - set(d2))
    only2 = sorted(set(d2) - set(d1))
    if only1 or only2:
        print('Relpaths mismatch!')
        print('Only in run1:', only1[:10])
        print('Only in run2:', only2[:10])
        # continue to report common items

    relpaths = sorted(set(d1.keys()) & set(d2.keys()))
    meta_diff = []
    wav_diff = []

    crit = ["seed_key","seed","label","category","drone_source","noise_sources",
            "doppler_shift_semitones","normalization_gain_db","peak_dbfs","rms_dbfs",
            "target_snr_db","actual_snr_db_exported","actual_snr_db_preexport"]

    for rp in relpaths:
        a = norm_meta(d1[rp])
        b = norm_meta(d2[rp])

        for k in crit:
            if a.get(k) != b.get(k):
                meta_diff.append((rp, k, a.get(k), b.get(k)))
                break

        p1 = run1_dir / rp
        p2 = run2_dir / rp
        if p1.exists() and p2.exists():
            h1, h2 = sha256_pcm_wav(p1), sha256_pcm_wav(p2)
            if h1 != h2:
                wav_diff.append((rp, h1[:16], h2[:16]))
        else:
            wav_diff.append((rp, 'MISSING' if not p1.exists() else '', 'MISSING' if not p2.exists() else ''))

    print(f'Total samples: {len(relpaths)}')
    print(f'Metadata first-diff count: {len(meta_diff)}')
    print(f'WAV hash diff count: {len(wav_diff)}')

    if meta_diff:
        print('\nFirst metadata diffs:')
        for rp,k,v1,v2 in meta_diff[:max_print]:
            print(f'  {rp}  field={k}  run1={v1}  run2={v2}')

    if wav_diff:
        print('\nFirst WAV hash diffs:')
        for rp,h1,h2 in wav_diff[:max_print]:
            print(f'  {rp}  {h1}.. != {h2}..')


if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Compare two augmentation runs (WAVs + metadata)")
    parser.add_argument('run1_dir', type=str, help='First run directory')
    parser.add_argument('run2_dir', type=str, help='Second run directory')
    parser.add_argument('--meta', type=str, default=None, help='Metadata JSONL filename (default: autodetect)')
    args = parser.parse_args()
    main(args.run1_dir, args.run2_dir, meta_name=args.meta)
