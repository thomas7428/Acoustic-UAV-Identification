#!/usr/bin/env python3
"""Run a small dataset generation with RIR enabled and perform sanity checks.

Creates a temporary small RIR bank if none exists, generates ~50 samples,
and checks metadata/WAV consistency, clipping, determinism and simple HF vs distance trend.
"""
import json
from pathlib import Path
import tempfile
import shutil
import numpy as np
import soundfile as sf
import os
import sys

from augment_dataset_v4 import run_smoke
from tools.compare_runs import load_jsonl, sha256_pcm_wav, norm_meta


def make_synthetic_rirs(out_dir: Path, sr=22050, count=6, dur_s=0.5, seed=123):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(count):
        t = np.linspace(0, dur_s, int(dur_s * sr), endpoint=False)
        # decaying noise impulse
        decay = np.exp(-t * (1.0 + i * 2.0))
        noise = rng.normal(scale=1.0, size=t.shape) * decay
        x = noise.astype('float32')
        p = out_dir / f'rir_{i:03d}.wav'
        sf.write(str(p), x, sr)
    return out_dir


def hf_ratio(x, sr, cutoff=5000.0):
    # compute energy ratio above cutoff
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    total = float(np.sum(np.abs(X) ** 2) + 1e-12)
    hf = float(np.sum(np.abs(X[freqs >= cutoff]) ** 2) + 1e-12)
    return hf / total


def run(strict=False, max_leak_auc=0.6, max_clip_count=0, min_unique_rir=3, min_unique_scene=3, max_stem_dominance=0.5):
    base_cfg = Path('0 - DADS dataset extraction/augment_config_v4.json')
    if not base_cfg.exists():
        print('Base config not found:', base_cfg)
        sys.exit(2)

    tmp = Path('tmp_dataset_sanity')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    tmp_rir_dir = tmp / 'rirs'
    make_synthetic_rirs(tmp_rir_dir, sr=22050, count=6)

    # make synthetic noise pool for scene mixer
    tmp_noise = tmp / 'noise_pool'
    tmp_noise.mkdir()
    rng = np.random.default_rng(12345)
    for i in range(8):
        x = (rng.normal(scale=0.2, size=int(0.8 * 22050)) * np.hanning(int(0.8 * 22050))).astype('float32')
        sf.write(str(tmp_noise / f'noise_{i:03d}.wav'), x, 22050)
    # make some hard negatives (engine-like)
    tmp_hard = tmp / 'hard_negatives'
    tmp_hard.mkdir()
    for i in range(3):
        t = np.linspace(0, 0.8, int(0.8 * 22050), endpoint=False)
        tone = np.sin(2 * np.pi * (200 + i * 50) * t) * np.exp(-t * 1.5)
        sf.write(str(tmp_hard / f'hard_{i:03d}.wav'), tone.astype('float32'), 22050)

    cfg = json.loads(base_cfg.read_text(encoding='utf-8'))
    # enable propagation and RIR with small bank
    cfg.setdefault('propagation', {})
    cfg['propagation']['enabled'] = True
    cfg['propagation']['rir'] = {
        'enabled': True,
        'rir_dir': str(tmp_rir_dir.resolve()),
        'select': {'strategy': 'rng'},
        'dry_wet': [0.2, 0.5],
        'normalize_rir': True
    }
    # enable scene mixer
    cfg['scene'] = {
        'enabled': True,
        'min_stems': 1,
        'max_stems': 4,
        'env_segments': 4,
        'env_activity_prob': 0.7,
        'gain_db_min': -6.0,
        'gain_db_max': -1.0,
        'noise_pool_dirs': [str(tmp_noise.resolve())]
    }
    # hard negatives config
    cfg.setdefault('no_drone_augmentation', {})
    cfg['no_drone_augmentation']['hard_negative_prob'] = 0.3
    cfg['no_drone_augmentation']['hard_negative_dir'] = str(tmp_hard.resolve())
    # small run: ~50 samples
    cfg['output']['samples_per_category_drone'] = 5
    cfg['output']['samples_per_category_no_drone'] = 5
    cfg['advanced']['random_seed'] = 999

    cfg_path = tmp / 'sanity_config.json'
    cfg_path.write_text(json.dumps(cfg), encoding='utf-8')

    run1 = tmp / 'run1'
    run2 = tmp / 'run2'

    print('Generating run1...')
    run_smoke(str(cfg_path), str(run1), dry_run=False)
    print('Generating run2...')
    run_smoke(str(cfg_path), str(run2), dry_run=False)

    def _load_meta_run(run_dir):
        # prefer root-level jsonl if present (legacy), otherwise concatenate per-split jsonls
        root = run_dir / 'augmentation_samples.jsonl'
        if root.exists():
            return load_jsonl(root)
        metas = []
        for s in ['train', 'val', 'test', 'report']:
            p = run_dir / s / 'augmentation_samples.jsonl'
            if p.exists():
                metas.extend(load_jsonl(p))
        return metas

    meta1 = _load_meta_run(run1)
    meta2 = _load_meta_run(run2)
    wavs = list((run1).rglob('*.wav'))
    print('JSONL lines:', len(meta1), 'WAV files:', len(wavs))
    if len(meta1) != len(wavs):
        print('FAIL: JSONL lines != WAV count')
        return 2

    # clipping check: look for clip_count in train_meta
    clip_issues = [m for m in meta1 if (m.get('train_meta', {}) or {}).get('clip_count', 0) > max_clip_count]
    if clip_issues:
        print('FAIL: clipping detected in samples:', len(clip_issues))
        if strict:
            return 2

    # determinism: compare pcm hashes (use train_meta.relpath)
    rels = [m.get('train_meta', {}).get('relpath') or m.get('train_meta', {}).get('filename') for m in meta1]
    pcm_diffs = 0
    for rp in rels:
        p1 = run1 / rp
        p2 = run2 / rp
        if p1.exists() and p2.exists():
            if sha256_pcm_wav(p1) != sha256_pcm_wav(p2):
                pcm_diffs += 1
        else:
            pcm_diffs += 1
    print('PCM diffs:', pcm_diffs)
    if pcm_diffs != 0:
        print('FAIL: determinism check failed (PCM diffs != 0)')
        return 2

    # rir_id distribution (from debug_meta)
    rir_ids = set([m.get('debug_meta', {}).get('rir_id') for m in meta1 if m.get('debug_meta', {}).get('rir_id') is not None])
    print('Unique RIR ids:', len(rir_ids))
    if len(rir_ids) < min_unique_rir and len(list(tmp_rir_dir.glob('*.wav'))) >= min_unique_rir:
        print(f'FAIL: RIR id distribution degenerate (<{min_unique_rir})')
        if strict:
            return 2

    # HF ratio vs distance (derive distance from category name if present)
    by_dist = {}
    for m in meta1:
        cat = (m.get('train_meta', {}) or {}).get('category', '')
        # parse number in category like 'drone_300m'
        d = None
        import re
        mo = re.search(r"(\d{2,4})m", cat)
        if mo:
            d = int(mo.group(1))
        else:
            continue
        p = run1 / ((m.get('train_meta', {}) or {}).get('relpath') or (m.get('train_meta', {}) or {}).get('filename'))
        if not p.exists():
            continue
        x, sr = sf.read(p, dtype='float32')
        r = hf_ratio(np.asarray(x, dtype=np.float32), sr)
        by_dist.setdefault(d, []).append(r)

    if by_dist:
        avgs = sorted([(d, float(np.mean(v))) for d, v in by_dist.items()])
        avgs_sorted = [v for (_, v) in avgs]
        # check non-increasing HF energy with distance (loose)
        decreasing = all(x >= y - 1e-6 for x, y in zip(avgs_sorted, avgs_sorted[1:]))
        print('HF avg by distance:', avgs)
        if not decreasing:
            print('WARN: HF ratio not strictly decreasing by distance (loose check)')

    # Scene diversity: require multiple scene_ids (from debug_meta)
    scene_ids = [ (m.get('debug_meta', {}) or {}).get('scene_id') for m in meta1 if (m.get('debug_meta', {}) or {}).get('scene_id')]
    uniq_scene = set(scene_ids)
    print('Unique scene ids:', len(uniq_scene))
    if len(uniq_scene) < min_unique_scene and len(list(tmp_noise.glob('*.wav'))) >= min_unique_scene:
        print(f'FAIL: scene diversity degenerate (<{min_unique_scene})')
        if strict:
            return 2

    # source dominance: no single stem id > X% frequency
    stem_counts = {}
    for m in meta1:
        for s in (m.get('stems') or []):
            stem_counts[s] = stem_counts.get(s, 0) + 1
    total_stems = sum(stem_counts.values()) if stem_counts else 0
    max_share = max((c / total_stems for c in stem_counts.values()), default=0.0)
    print('Top stem share:', max_share)
    if max_share > max_stem_dominance:
        print(f'FAIL/WARN: single stem accounts for >{max_stem_dominance*100:.0f}% of stems')
        if strict:
            return 2

    # metadata-only leak check: use train_meta only, 5-fold CV, include missingness features
    try:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        # collect train_meta dicts
        train_dicts = [m.get('train_meta', {}) for m in meta1]
        labels = [int(m.get('train_meta', {}).get('label', 0)) for m in meta1]

        # build feature set: whitelist safe train fields (strict: exclude distance/target which moved to debug_meta)
        safe_keys = ['actual_snr_db_preexport', 'peak_dbfs', 'rms_dbfs', 'clip_count']
        feat_dicts = []
        for td in train_dicts:
            fd = {}
            for k in safe_keys:
                v = td.get(k)
                # missingness feature
                fd[f'has_{k}'] = 0 if v is None else 1
                if v is None:
                    continue
                if isinstance(v, (int, float, bool)):
                    fd[k] = v
                else:
                    fd[k] = str(v)
            feat_dicts.append(fd)

        vec = DictVectorizer(sparse=False)
        X = vec.fit_transform(feat_dicts)

        if X.shape[0] >= 10 and X.shape[1] > 0:
            # choose n_splits safely based on class counts
            from collections import Counter
            cls_counts = Counter(labels)
            min_class_count = min(cls_counts.values()) if cls_counts else 0
            if min_class_count < 2:
                print('Not enough members in each class for CV; skipping leak CV')
                mean_auc = float('nan')
            else:
                n_splits = min(5, max(2, min_class_count))
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                aucs = []
                coef_accum = {f: [] for f in vec.get_feature_names_out()}
                for tr_idx, te_idx in skf.split(X, labels):
                    Xtr, Xte = X[tr_idx], X[te_idx]
                    ytr = [labels[i] for i in tr_idx]
                    yte = [labels[i] for i in te_idx]
                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(Xtr, ytr)
                    probs = clf.predict_proba(Xte)[:, 1]
                    auc = float(roc_auc_score(yte, probs))
                    aucs.append(auc)
                    # record coefficients
                    for fn, c in zip(vec.get_feature_names_out(), clf.coef_.ravel()):
                        coef_accum[fn].append(abs(float(c)))
                mean_auc = float(np.mean(aucs))
            print('Metadata-only CV AUC (5-fold mean):', mean_auc)
            # compute mean absolute coeffs and print top-20
            mean_coefs = {fn: float(np.mean(vals)) for fn, vals in coef_accum.items() if vals}
            top_feats = sorted(mean_coefs.items(), key=lambda x: x[1], reverse=True)[:20]
            print('\nTop-20 predictive metadata features (mean abs coef):')
            for fn, val in top_feats:
                print(f'  {fn}: {val:.6f}')

            if mean_auc > max_leak_auc:
                print(f'WARN: metadata-only classifier AUC > {max_leak_auc} (possible leak)')
                if strict:
                    return 2
        else:
            print('Not enough features/samples for metadata leak CV check')
    except Exception as e:
        print('Skipping metadata leak check (sklearn missing or error):', e)

    print('RUN_V4_DATASET_SANITY: OK')
    return 0


def validate_dataset(dataset_path: Path, strict=False, max_leak_auc=0.6, max_clip_count=0, min_unique_rir=3, min_unique_scene=3, max_stem_dominance=0.5):
    """Validate an existing dataset folder structured as datasets/<id>/<split>/*"""
    ds = Path(dataset_path)
    if not ds.exists():
        print('Dataset not found:', ds)
        return 2

    # defensive: fail/warn if a root-level augmentation_samples.jsonl exists alongside split jsonls
    root_meta = ds / 'augmentation_samples.jsonl'
    if root_meta.exists():
        # check for any split-contained jsonl
        splits_with_meta = [d for d in ds.iterdir() if d.is_dir() and (d / 'augmentation_samples.jsonl').exists()]
        if splits_with_meta:
            msg = f'ERROR: Found root-level augmentation_samples.jsonl alongside split jsonl files in {ds} (ambiguous).'
            print(msg)
            if strict:
                return 2
            else:
                print('Warning: consider removing root augmentation_samples.jsonl; proceeding with split-contained validation')

    # iterate splits
    splits = [d for d in ds.iterdir() if d.is_dir()]
    all_train_metas = []
    for s in sorted(splits):
        meta_path = s / 'augmentation_samples.jsonl'
        metas = []
        if meta_path.exists():
            with meta_path.open('r', encoding='utf-8') as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    metas.append(json.loads(line))
        wavs = list(s.rglob('*.wav'))
        print(f"Split {s.name}: JSONL lines: {len(metas)} WAV files: {len(wavs)}")
        if len(metas) != len(wavs):
            print(f'FAIL: JSONL lines != WAV count in split {s.name}')
            if strict:
                return 2
        # clipping check
        clip_issues = [m for m in metas if (m.get('train_meta', {}) or {}).get('clip_count', 0) > max_clip_count]
        if clip_issues:
            print(f'FAIL: clipping detected in split {s.name}:', len(clip_issues))
            if strict:
                return 2
        if s.name == 'train':
            all_train_metas.extend(metas)

    # metadata-only leak CV on train split
    if all_train_metas:
        try:
            from sklearn.feature_extraction import DictVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import StratifiedKFold
            train_dicts = [m.get('train_meta', {}) for m in all_train_metas]
            labels = [int(m.get('train_meta', {}).get('label', 0)) for m in all_train_metas]
            safe_keys = ['actual_snr_db_preexport', 'peak_dbfs', 'rms_dbfs', 'clip_count']
            feat_dicts = []
            for td in train_dicts:
                fd = {}
                for k in safe_keys:
                    v = td.get(k)
                    fd[f'has_{k}'] = 0 if v is None else 1
                    if v is None:
                        continue
                    if isinstance(v, (int, float, bool)):
                        fd[k] = v
                    else:
                        fd[k] = str(v)
                feat_dicts.append(fd)
            vec = DictVectorizer(sparse=False)
            X = vec.fit_transform(feat_dicts)
            mean_auc = float('nan')
            if X.shape[0] >= 10 and X.shape[1] > 0:
                from collections import Counter
                cls_counts = Counter(labels)
                min_class_count = min(cls_counts.values()) if cls_counts else 0
                if min_class_count >= 2:
                    n_splits = min(5, max(2, min_class_count))
                    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    aucs = []
                    for tr_idx, te_idx in skf.split(X, labels):
                        Xtr, Xte = X[tr_idx], X[te_idx]
                        ytr = [labels[i] for i in tr_idx]
                        yte = [labels[i] for i in te_idx]
                        clf = LogisticRegression(max_iter=1000)
                        clf.fit(Xtr, ytr)
                        probs = clf.predict_proba(Xte)[:, 1]
                        aucs.append(float(roc_auc_score(yte, probs)))
                    mean_auc = float(np.mean(aucs))
            print('Metadata-only CV AUC (train split):', mean_auc)
            if mean_auc > max_leak_auc:
                print(f'WARN: metadata-only classifier AUC > {max_leak_auc} (possible leak)')
                if strict:
                    return 2
        except Exception as e:
            print('Skipping metadata leak check (sklearn missing or error):', e)

    print('VALIDATE_V4_DATASET: OK')
    return 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='Path to existing dataset directory to validate')
    parser.add_argument('--strict', action='store_true', help='Fail on warnings')
    parser.add_argument('--max_leak_auc', type=float, default=0.6)
    parser.add_argument('--max_clip_count', type=int, default=0)
    parser.add_argument('--min_unique_rir', type=int, default=3)
    parser.add_argument('--min_unique_scene', type=int, default=3)
    parser.add_argument('--max_stem_dominance', type=float, default=0.5)
    args = parser.parse_args()
    if args.dataset:
        sys.exit(validate_dataset(Path(args.dataset), strict=args.strict, max_leak_auc=args.max_leak_auc, max_clip_count=args.max_clip_count, min_unique_rir=args.min_unique_rir, min_unique_scene=args.min_unique_scene, max_stem_dominance=args.max_stem_dominance))
    sys.exit(run(strict=args.strict, max_leak_auc=args.max_leak_auc, max_clip_count=args.max_clip_count, min_unique_rir=args.min_unique_rir, min_unique_scene=args.min_unique_scene, max_stem_dominance=args.max_stem_dominance))
