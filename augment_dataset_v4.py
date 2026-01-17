#!/usr/bin/env python3
"""Wrapper entry for v4 generation.

This script is the top-level orchestrator. It prefers the local copy of the
augmentation package placed under `0 - DADS dataset extraction/augment/` so
all augmentation logic and modifiers live under the extraction root.
"""
import json
from pathlib import Path
import argparse
import sys

# Prefer local extraction-root augment package so imports resolve to the
# canonical location under '0 - DADS dataset extraction'.
PROJ_ROOT = Path(__file__).resolve().parent
EXTRACTION = PROJ_ROOT / '0 - DADS dataset extraction'
if str(EXTRACTION) not in sys.path:
    sys.path.insert(0, str(EXTRACTION))

from augment.v4.writer import SplitWriter
from augment.v4.rng import rng_for_key, seed_from_key
from augment.v4.io import write_wav, append_jsonl_line, truncate_jsonl, write_summary
from augment.v4.schema import SampleMeta
from augment.v4.pipeline import Pipeline
from augment.v4.transforms import source_transform, mix_transform, hardware_transform, post_transform
from augment.v4.transforms import distance_attenuation_transform, air_absorption_lpf_transform, rir_convolution_transform, scene_mix_transform
import multiprocessing
import os as _os


def _generate_task(args_tuple):
    """Top-level worker for multiprocessing: args_tuple contains the parameters needed."""
    task, cfg, master_seed, sr, duration_s, drone_files, no_drone_files, no_drone_dir, _drone_snr_pool, _drone_distance_pool = args_tuple
    try:
        from augment.v4.rng import rng_for_key, seed_from_key
        from augment.v4.transforms import source_transform, mix_transform, hardware_transform, post_transform
        from augment.v4.transforms import distance_attenuation_transform, air_absorption_lpf_transform, rir_convolution_transform, scene_mix_transform
        import numpy as _np
    except Exception as e:
        return {'error': str(e)}

    task_kind = task.get('kind')
    category = task.get('category')
    i = task.get('index')
    seed_key = task.get('seed_key')
    rng = rng_for_key(master_seed, seed_key)

    def sample_distance_for_category(rng, category, propagation_cfg):
        if isinstance(category, dict) and category.get('distance_m') is not None:
            return float(category.get('distance_m'))
        if isinstance(category, dict) and category.get('distance_min') is not None and category.get('distance_max') is not None:
            return float(rng.uniform(float(category.get('distance_min')), float(category.get('distance_max'))))
        if isinstance(category, dict) and category.get('distance_bounds'):
            b = category.get('distance_bounds')
            return float(rng.uniform(float(b[0]), float(b[1])))
        if cfg.get('propagation', {}).get('distance_bounds'):
            b = cfg.get('propagation', {}).get('distance_bounds')
            return float(rng.uniform(float(b[0]), float(b[1])))
        return None

    try:
        if task_kind == 'drone':
            cat_name = category['name']
            drone_file = rng.choice(drone_files) if drone_files else None
            sample_spec = {'type': 'drone', 'drone_file': drone_file, 'noise_files': [], 'duration_s': duration_s}
            audio, meta_delta = source_transform(sample_spec, sr, rng, {})
            scene_cfg = cfg.get('scene', {})
            if scene_cfg.get('enabled', False):
                noise_pool = scene_cfg.get('noise_pool_dirs') or [str(no_drone_dir)]
                hard_neg_cfg = cfg.get('no_drone_augmentation', {})
                hard_negative_used = False
                if hard_neg_cfg.get('hard_negative_prob', 0.0) > 0.0 and hard_neg_cfg.get('hard_negative_dir'):
                    if rng.random() < float(hard_neg_cfg.get('hard_negative_prob', 0.0)):
                        noise_pool = [hard_neg_cfg.get('hard_negative_dir')] + list(noise_pool)
                        hard_negative_used = True
                scene_audio, scene_meta = scene_mix_transform(None, sr, rng, {}, {**scene_cfg, 'duration_s': duration_s}, noise_pool)
            else:
                scene_audio = _np.zeros(int(sr * duration_s), dtype=_np.float32)
                scene_meta = {}
                hard_negative_used = False
            target_snr_db = float(category.get('snr_db', category.get('target_snr_db', 0)))
            rms_scene = _np.sqrt(float(_np.mean(_np.square(scene_audio)))) if scene_audio.size else 0.0
            rms_drone = _np.sqrt(float(_np.mean(_np.square(audio)))) if audio.size else 1e-12
            if rms_scene <= 0:
                scale = 1.0
            else:
                desired = 10.0 ** (target_snr_db / 20.0)
                scale = (rms_scene * desired) / max(rms_drone, 1e-12)
            drone_scaled = audio * scale
            mixed = scene_audio + drone_scaled
            mix_meta = {**scene_meta, 'hard_negative_used': bool(hard_negative_used)}
            propagation_cfg = cfg.get('propagation', {})
            if propagation_cfg.get('enabled', False):
                distance_m = sample_distance_for_category(rng, category, propagation_cfg)
                base_beta = propagation_cfg.get('beta', 0.5)
                beta_range = propagation_cfg.get('beta_jitter') or propagation_cfg.get('beta_range')
                if beta_range and rng is not None:
                    beta = float(rng.uniform(float(beta_range[0]), float(beta_range[1])))
                else:
                    beta = float(base_beta)
                mixed, att_meta = distance_attenuation_transform(mixed, sr, rng, {'category': cat_name}, distance_m=distance_m, alpha=propagation_cfg.get('alpha', 1.0), ref_distance=propagation_cfg.get('ref_distance', 1.0))
                mixed, lpf_meta = air_absorption_lpf_transform(mixed, sr, rng, {'category': cat_name}, distance_m=distance_m, base_fc=propagation_cfg.get('base_fc', 8000.0), beta=beta, ref_distance=propagation_cfg.get('ref_distance', 1.0), min_fc=propagation_cfg.get('min_fc', 500.0))
                rir_cfg = propagation_cfg.get('rir', {})
                if rir_cfg.get('enabled', False):
                    mixed, rir_meta = rir_convolution_transform(mixed, sr, rng, {'category': cat_name}, rir_cfg, distance_m=distance_m)
                else:
                    rir_meta = {}
            else:
                att_meta = {}
                lpf_meta = {}
                rir_meta = {}
            hw_audio, hw_meta = hardware_transform(mixed, sr, rng, {}, cfg.get('mems_simulation', {}))
            final_audio, post_meta = post_transform(hw_audio, sr, rng, {}, cfg.get('audio_parameters', {}))
            filename = f"uav_{cat_name}_{i}.wav"
            seed = int(seed_from_key(master_seed, seed_key))
            relpath = None
            extras = {}
            for d in (meta_delta, mix_meta, att_meta, lpf_meta, rir_meta, hw_meta, post_meta):
                for k, v in (d or {}).items():
                    if k == 'noise_buffers':
                        continue
                    try:
                        if isinstance(v, (_np.floating, _np.integer)):
                            extras[k] = float(v)
                            continue
                    except Exception:
                        pass
                    if hasattr(v, 'tolist') and not isinstance(v, dict):
                        continue
                    extras[k] = v
            distance_m = category.get('distance_m') if isinstance(category, dict) else None
            target_snr_val = float(category.get('snr_db', category.get('target_snr_db', None)))
            train_meta = {'relpath': relpath, 'seed_key': seed_key, 'seed': seed, 'label': 1, 'category': cat_name, 'actual_snr_db_preexport': extras.get('actual_snr_db_preexport'), 'peak_dbfs': extras.get('peak_dbfs'), 'rms_dbfs': extras.get('rms_dbfs'), 'clip_count': int(extras.get('clip_count', 0))}
            debug_meta = {**extras, 'drone_source': str(drone_file) if drone_file is not None else None, 'mix_meta': mix_meta, 'distance_m': distance_m, 'target_snr_db': target_snr_val}
            return {'label': '1', 'filename': filename, 'audio': final_audio, 'sr': sr, 'train_meta': train_meta, 'debug_meta': debug_meta, 'seed_key': seed_key, 'drone_file': str(drone_file) if drone_file is not None else None, 'assigned_split': task.get('split', 'train')}

        else:
            # no-drone
            cat_name = category['name']
            noise_files = rng.choice(no_drone_files, size=min(int(category.get('num_noise_sources', category.get('num_background_noises',1))), len(no_drone_files)), replace=False).tolist() if no_drone_files else []
            scene_cfg = cfg.get('scene', {})
            if scene_cfg.get('enabled', False):
                noise_pool = scene_cfg.get('noise_pool_dirs') or [str(no_drone_dir)]
                hard_neg_cfg = category if category.get('hard_negative_dir') else cfg.get('no_drone_augmentation', {})
                hard_negative_used = False
                if hard_neg_cfg.get('hard_negative_prob', 0.0) > 0.0 and hard_neg_cfg.get('hard_negative_dir'):
                    if rng.random() < float(hard_neg_cfg.get('hard_negative_prob', 0.0)):
                        noise_pool = [hard_neg_cfg.get('hard_negative_dir')] + list(noise_pool)
                        hard_negative_used = True
                scene_audio, scene_meta = scene_mix_transform(None, sr, rng, {}, {**scene_cfg, 'duration_s': duration_s}, noise_pool)
                mixed = scene_audio
                mix_meta = {**scene_meta, 'hard_negative_used': bool(hard_negative_used)}
            else:
                sample_spec = {'type': 'no_drone', 'noise_files': noise_files, 'duration_s': duration_s}
                audio, meta_delta = source_transform(sample_spec, sr, rng, {})
                merged_meta = {**meta_delta, 'class': 'no_drone'}
                mixed, mix_meta = mix_transform(audio, sr, rng, merged_meta, category, cfg)
            hw_audio, hw_meta = hardware_transform(mixed, sr, rng, {}, cfg.get('mems_simulation', {}))
            final_audio, post_meta = post_transform(hw_audio, sr, rng, {}, cfg.get('audio_parameters', {}))
            filename = f"bg_{cat_name}_{i}.wav"
            seed = int(seed_from_key(master_seed, seed_key))
            relpath = None
            extras = {}
            for d in (meta_delta, mix_meta, att_meta, lpf_meta, hw_meta, post_meta):
                for k, v in (d or {}).items():
                    if k == 'noise_buffers':
                        continue
                    try:
                        if isinstance(v, (_np.floating, _np.integer)):
                            extras[k] = float(v)
                            continue
                    except Exception:
                        pass
                    if hasattr(v, 'tolist') and not isinstance(v, dict):
                        continue
                    extras[k] = v
            distance_m = category.get('distance_m') if isinstance(category, dict) else None
            sampled_target_snr = None
            sampled_distance = None
            if _drone_snr_pool:
                sampled_target_snr = float(rng.choice(_drone_snr_pool))
            if (_drone_distance_pool) and (distance_m is None):
                sampled_distance = int(rng.choice(_drone_distance_pool))
            train_meta = {'relpath': relpath, 'seed_key': seed_key, 'seed': seed, 'label': 0, 'category': cat_name, 'actual_snr_db_preexport': extras.get('actual_snr_db_preexport'), 'peak_dbfs': extras.get('peak_dbfs'), 'rms_dbfs': extras.get('rms_dbfs'), 'clip_count': int(extras.get('clip_count', 0))}
            debug_meta = {**extras, 'hard_negative_used': mix_meta.get('hard_negative_used') if isinstance(mix_meta, dict) else None, 'distance_m': sampled_distance if sampled_distance is not None else distance_m, 'target_snr_db': sampled_target_snr}
            return {'label': '0', 'filename': filename, 'audio': final_audio, 'sr': sr, 'train_meta': train_meta, 'debug_meta': debug_meta, 'seed_key': seed_key, 'drone_file': None, 'assigned_split': task.get('split', 'train')}
    except Exception as e:
        return {'error': str(e)}

import numpy as np
import time
from datetime import timezone

def run_smoke(config_path, out_dir, dry_run=False, total=None, drone_count=None, no_drone_count=None, split_ratios=None):
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # dataset layout: expected out is datasets/<dataset_id>
    # load splits.json if present
    splits_map = {}
    splits_path = out / 'splits.json'
    if splits_path.exists():
        try:
            splits_map = json.loads(splits_path.read_text(encoding='utf-8'))
        except Exception:
            splits_map = {}

    # writers cache per split
    writers = {}

    def get_writer_for_sample(label, drone_file_path, seed_key, override_split=None):
        # determine split: if drone source present, lookup by stem; else fallback to 'train'
        split = 'train'
        if override_split:
            split = override_split
        elif drone_file_path:
            src = Path(drone_file_path).stem
            split = splits_map.get(src, 'train')
        # map legacy split names to canonical dataset_* layout
        canonical = split
        if not split.startswith('dataset_'):
            canonical = f'dataset_{split}'
        # create writer if missing
        if canonical not in writers:
            writers[canonical] = SplitWriter(out / canonical)
        return writers[canonical], canonical

    sr = cfg.get('audio_parameters', {}).get('sample_rate', 16000)
    duration_s = float(cfg.get('audio_parameters', {}).get('target_duration_sec', cfg.get('audio_parameters', {}).get('duration_s', 1.0)))

    # discover source files similar to v3
    import config as project_config
    offline_dir = Path(project_config.DATASET_DADS_OFFLINE_DIR)
    drone_dir = offline_dir / '1'
    no_drone_dir = offline_dir / '0'
    # pass simple strings to worker args to avoid heavy pickling
    drone_files = sorted([str(p) for p in drone_dir.glob('*.wav')]) if drone_dir.exists() else []
    no_drone_files = sorted([str(p) for p in no_drone_dir.glob('*.wav')]) if no_drone_dir.exists() else []

    master_seed = int(cfg.get('advanced', {}).get('random_seed', 0))

    # precompute drone SNR and distance pools for symmetric metadata sampling
    import re
    drone_categories = cfg.get('drone_augmentation', {}).get('categories', [])
    _drone_snr_pool = [float(c.get('snr_db', 0.0)) for c in drone_categories if c.get('snr_db') is not None]
    _drone_distance_pool = []
    for c in drone_categories:
        mo = re.search(r"(\d{2,4})m", str(c.get('name', '')))
        if mo:
            _drone_distance_pool.append(int(mo.group(1)))

    total_generated = 0
    samples_meta = []

    # determine worker parallelism (env override -> config -> cpu_count)
    env_workers = int(_os.environ.get('AUGMENTATION_WORKERS', 0))
    cfg_workers = int(getattr(project_config, 'AUGMENTATION_MAX_WORKERS', 0) or 0)
    max_workers = env_workers if env_workers > 0 else (cfg_workers if cfg_workers > 0 else 1)

    # Build task list for all samples (drone + no_drone)
    tasks = []
    # determine counts per class (config JSON takes precedence over CLI)
    cfg_out = cfg.get('output', {})
    # split ratios: prefer config.validation_strategy.split_ratio or cfg.output.split_ratio
    cfg_split = None
    vs_split = cfg.get('validation_strategy', {}).get('split_ratio')
    if vs_split is not None:
        if isinstance(vs_split, (list, tuple)):
            cfg_split = tuple(vs_split)
        elif isinstance(vs_split, (int, float)):
            # single number interpreted as train ratio
            tr = float(vs_split)
            rest = max(0.0, 1.0 - tr)
            cfg_split = (tr, rest / 2.0, rest / 2.0)
    elif cfg_out.get('split_ratio'):
        cfg_split = tuple(cfg_out.get('split_ratio'))
    if cfg_split is not None:
        split_ratios = cfg_split
    if split_ratios is None:
        split_ratios = (0.7, 0.15, 0.15)

    # totals: precedence order -> cfg.output.total_per_class OR cfg.output.total_samples OR CLI args
    total_cfg_per_class = cfg_out.get('total_per_class')
    total_cfg_total = cfg_out.get('total_samples')
    if total_cfg_per_class:
        drone_count = int(total_cfg_per_class.get('drone', 0))
        no_drone_count = int(total_cfg_per_class.get('no_drone', 0))
    elif total_cfg_total is not None:
        # split evenly unless cfg specifies class proportions
        drone_count = int(total_cfg_total // 2)
        no_drone_count = int(total_cfg_total - drone_count)
    else:
        # fallback to legacy per-category base counts if CLI not provided
        if drone_count is None and cfg_out.get('samples_per_category_drone'):
            base = int(cfg_out.get('samples_per_category_drone', 1))
            drone_count = base * max(1, len(cfg.get('drone_augmentation', {}).get('categories', [])))
        if no_drone_count is None and cfg_out.get('samples_per_category_no_drone'):
            base2 = int(cfg_out.get('samples_per_category_no_drone', 1))
            no_drone_count = base2 * max(1, len(cfg.get('no_drone_augmentation', {}).get('categories', [])))
        if total is not None and (drone_count is None and no_drone_count is None):
            drone_count = int(total // 2)
            no_drone_count = int(total - drone_count)
    # helper: allocate integer sample counts per category so totals match exactly
    def _allocate_counts(categories, total):
        if not categories:
            return {}
        props = [float(c.get('proportion', 1.0)) for c in categories]
        s = sum(props) or 1.0
        norms = [p / s for p in props]
        float_counts = [n * total for n in norms]
        base = [int(fc) for fc in float_counts]
        rem = total - sum(base)
        # distribute remainder by largest fractional parts
        fracs = [(i, float_counts[i] - base[i]) for i in range(len(base))]
        fracs.sort(key=lambda x: x[1], reverse=True)
        for i in range(rem):
            idx = fracs[i % len(fracs)][0]
            base[idx] += 1
        return {categories[i]['name']: base[i] for i in range(len(categories))}

    # allocate per-category counts exactly
    drone_cats = cfg.get('drone_augmentation', {}).get('categories', [])
    no_drone_cats = cfg.get('no_drone_augmentation', {}).get('categories', [])
    drone_alloc = _allocate_counts(drone_cats, int(drone_count or 0))
    no_drone_alloc = _allocate_counts(no_drone_cats, int(no_drone_count or 0))

    # deterministic RNG for split assignment
    master_seed = int(cfg.get('advanced', {}).get('random_seed', 0))
    split_rng = np.random.default_rng(master_seed)

    # build drone tasks
    for category in drone_cats:
        cat_name = category['name']
        samples_count = int(drone_alloc.get(cat_name, 0))
        for i in range(samples_count):
            seed_key = f"{cat_name}|{i}"
            split_choice = split_rng.choice(['train', 'val', 'test'], p=list(split_ratios))
            tasks.append({'kind': 'drone', 'category': category, 'index': i, 'seed_key': seed_key, 'split': split_choice})

    # No-drone categories tasks
    # build no-drone tasks
    for category in no_drone_cats:
        cat_name = category['name']
        samples_count = int(no_drone_alloc.get(cat_name, 0))
        for i in range(samples_count):
            seed_key = f"{cat_name}|{i}"
            split_choice = split_rng.choice(['train', 'val', 'test'], p=list(split_ratios))
            tasks.append({'kind': 'no_drone', 'category': category, 'index': i, 'seed_key': seed_key, 'split': split_choice})

    # Prepare args list for worker function (top-level _generate_task)
    args_list = [
        (task, cfg, master_seed, sr, duration_s, drone_files, no_drone_files, no_drone_dir, _drone_snr_pool, _drone_distance_pool)
        for task in tasks
    ]

    # simple logger for stderr
    def _log(*a, **kw):
        print(*a, file=sys.stderr, **kw)

    # sanity check: expected total vs tasks
    expected_total = (int(drone_count or 0) + int(no_drone_count or 0))
    if expected_total and len(tasks) != expected_total:
        _log(f"ERROR: requested total {expected_total} but built {len(tasks)} tasks")
        raise RuntimeError(f"requested total {expected_total} != built tasks {len(tasks)}")

    # progress helpers
    def _print_progress(completed, success, total, start_ts):
        if total <= 0:
            return
        now = time.time()
        elapsed = now - start_ts
        per = elapsed / completed if completed > 0 else 0
        remaining = max(0, total - completed)
        eta = remaining * per if per > 0 else 0
        pct = (completed / total) * 100.0
        sys.stdout.write(f"\rProcessed {completed}/{total} ({pct:.1f}%) - generated={success} - elapsed={int(elapsed)}s ETA={int(eta)}s")
        sys.stdout.flush()

    # Execute tasks in parallel and serialize writes in main process
    if max_workers > 1 and len(tasks) > 0:
        start_ts = time.time()
        done = 0
        with multiprocessing.Pool(processes=min(max_workers, len(tasks))) as pool:
                for res in pool.imap_unordered(_generate_task, args_list):
                    try:
                        if not res or res.get('error'):
                            _log('worker error:', res and res.get('error'))
                            continue
                        # determine split and write
                        writer, split = get_writer_for_sample(res['label'], res.get('drone_file'), res['seed_key'], override_split=res.get('assigned_split'))
                        if not dry_run:
                            writer.write_wav(res['label'], res['filename'], res['audio'], res['sr'])
                            # finalize relpath
                            relpath = f"{split}/{res['label']}/{res['filename']}"
                            res['train_meta']['relpath'] = relpath
                            writer.append_jsonl(res['train_meta'], res.get('debug_meta'))
                            samples_meta.append({'train_meta': res['train_meta'], 'debug_meta': res.get('debug_meta')})
                            total_generated += 1
                    except Exception as e:
                        _log('write error for', res and res.get('seed_key'), ':', e)
                    finally:
                        done += 1
                        if getattr(run_smoke, '__show_progress__', False):
                            _print_progress(done, total_generated, len(tasks), start_ts)
        if getattr(run_smoke, '__show_progress__', False):
            sys.stdout.write("\n")
    else:
        # fallback to sequential generation (previous logic preserved)
        start_ts = time.time()
        done = 0
        for args_tuple in args_list:
            try:
                res = _generate_task(args_tuple)
                if not res or res.get('error'):
                    _log('worker error (sequential):', res and res.get('error'))
                    continue
                writer, split = get_writer_for_sample(res['label'], res.get('drone_file'), res['seed_key'], override_split=res.get('assigned_split'))
                if not dry_run:
                    writer.write_wav(res['label'], res['filename'], res['audio'], res['sr'])
                    relpath = f"{split}/{res['label']}/{res['filename']}"
                    res['train_meta']['relpath'] = relpath
                    writer.append_jsonl(res['train_meta'], res.get('debug_meta'))
                    samples_meta.append({'train_meta': res['train_meta'], 'debug_meta': res.get('debug_meta')})
                    total_generated += 1
            except Exception as e:
                _log('write error (sequential) for', res and res.get('seed_key'), ':', e)
            finally:
                done += 1
                if getattr(run_smoke, '__show_progress__', False):
                    _print_progress(done, total_generated, len(tasks), start_ts)
        if getattr(run_smoke, '__show_progress__', False):
            sys.stdout.write("\n")

    

    # summary
    summary_obj = {
        'generation_time': None,
        'version': '4.0-migration-skeleton',
        'total_generated': total_generated,
        'samples': samples_meta
    }
    write_summary(Path(out) / 'augmentation_summary.json', summary_obj)

    # write provenance files expected by verifier
    try:
        import shutil
        from datetime import datetime
        # copy original config to build_config.json
        cfg_src = Path(config_path)
        if cfg_src.exists():
            shutil.copy2(cfg_src, out / 'build_config.json')
        # effective config (resolved)
        (out / 'effective_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        build_info = {
            'generator': 'augment_dataset_v4.py',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'pid': _os.getpid(),
            'version': cfg.get('version', 'unknown'),
        }
        (out / 'build_info.json').write_text(json.dumps(build_info, indent=2), encoding='utf-8')
        # ensure splits.json exists (write current splits_map)
        (out / 'splits.json').write_text(json.dumps(splits_map, indent=2), encoding='utf-8')
    except Exception:
        pass

    return total_generated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--progress', action='store_true', help='Show progress and ETA during generation')
    parser.add_argument('--total', type=int, default=None, help='Total samples to generate (overrides config when set)')
    parser.add_argument('--drone_count', type=int, default=None, help='Total drone-class samples to generate')
    parser.add_argument('--no_drone_count', type=int, default=None, help='Total no-drone samples to generate')
    parser.add_argument('--split_ratio', type=str, default=None, help='Comma-separated split ratios for train,val,test e.g. 0.7,0.15,0.15')
    args = parser.parse_args()
    # expose progress flag to run_smoke
    setattr(run_smoke, '__show_progress__', bool(args.progress))
    split_ratios = None
    if args.split_ratio:
        parts = [float(x) for x in args.split_ratio.split(',')]
        if len(parts) == 3 and abs(sum(parts) - 1.0) < 1e-6:
            split_ratios = tuple(parts)
    run_smoke(args.config, args.out_dir, dry_run=False, total=args.total, drone_count=args.drone_count, no_drone_count=args.no_drone_count, split_ratios=split_ratios)

if __name__ == '__main__':
    main()
