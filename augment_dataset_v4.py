#!/usr/bin/env python3
"""Minimal v4 entry script that reproduces v3 smoke using v4 pipeline wrappers.
"""
import json
from pathlib import Path
import argparse
from augment.v4.rng import rng_for_key, seed_from_key
from augment.v4.io import write_wav, append_jsonl_line, truncate_jsonl, write_summary
from augment.v4.schema import SampleMeta
from augment.v4.pipeline import Pipeline
from augment.v4.transforms import source_transform, mix_transform, hardware_transform, post_transform
from augment.v4.transforms import distance_attenuation_transform, air_absorption_lpf_transform, rir_convolution_transform, scene_mix_transform

import numpy as np

def run_smoke(config_path, out_dir, dry_run=False):
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # stream/summary paths
    stream = out / 'augmentation_samples.jsonl'
    summary = out / 'augmentation_summary.json'
    truncate_jsonl(stream)

    sr = cfg.get('audio_parameters', {}).get('sample_rate', 16000)
    duration_s = float(cfg.get('audio_parameters', {}).get('duration_s', 1.0))

    # discover source files similar to v3
    import config as project_config
    offline_dir = Path(project_config.DATASET_DADS_OFFLINE_DIR)
    drone_dir = offline_dir / '1'
    no_drone_dir = offline_dir / '0'
    drone_files = sorted(list(drone_dir.glob('*.wav'))) if drone_dir.exists() else []
    no_drone_files = sorted(list(no_drone_dir.glob('*.wav'))) if no_drone_dir.exists() else []

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

    # For v3 smoke config: iterate categories in drone and no_drone
    # We'll follow v3 structure: generate one sample per category entry as in smoke
    # Drone categories
    for category in cfg.get('drone_augmentation', {}).get('categories', []):
        cat_name = category['name']
        samples_count = max(1, int(cfg['output']['samples_per_category_drone'] * category.get('proportion', 1)))
        for i in range(samples_count):
            seed_key = f"{cat_name}|{i}"
            rng = rng_for_key(master_seed, seed_key)

            # Choose drone file and build scene if enabled
            drone_file = rng.choice(drone_files) if drone_files else None
            # load drone-only audio via source_transform (no background noises)
            sample_spec = {
                'type': 'drone',
                'drone_file': drone_file,
                'noise_files': [],
                'duration_s': duration_s
            }
            audio, meta_delta = source_transform(sample_spec, sr, rng, {})

            # Build scene (noise-only) if configured
            scene_cfg = cfg.get('scene', {})
            if scene_cfg.get('enabled', False):
                # noise pool dirs: prefer explicit config, otherwise use no_drone_dir
                noise_pool = scene_cfg.get('noise_pool_dirs') or [str(no_drone_dir)]
                # hard negatives handling
                hard_neg_cfg = cfg.get('no_drone_augmentation', {})
                hard_negative_used = False
                if hard_neg_cfg.get('hard_negative_prob', 0.0) > 0.0 and hard_neg_cfg.get('hard_negative_dir'):
                    if rng.random() < float(hard_neg_cfg.get('hard_negative_prob', 0.0)):
                        # prepend hard negative dir to pool so selection may include them
                        noise_pool = [hard_neg_cfg.get('hard_negative_dir')] + list(noise_pool)
                        hard_negative_used = True

                scene_audio, scene_meta = scene_mix_transform(None, sr, rng, {}, {**scene_cfg, 'duration_s': duration_s}, noise_pool)
            else:
                scene_audio = np.zeros(int(sr * duration_s), dtype=np.float32)
                scene_meta = {}

            # mix drone into scene at target SNR
            # obtain mixed via simple rms scaling to reach category['snr_db'] if present
            target_snr_db = float(category.get('snr_db', category.get('target_snr_db', 0)))
            # compute rms
            rms_scene = np.sqrt(float(np.mean(np.square(scene_audio)))) if scene_audio.size else 0.0
            rms_drone = np.sqrt(float(np.mean(np.square(audio)))) if audio.size else 1e-12
            if rms_scene <= 0:
                scale = 1.0
            else:
                desired = 10.0 ** (target_snr_db / 20.0)
                scale = (rms_scene * desired) / max(rms_drone, 1e-12)
            drone_scaled = audio * scale
            mixed = scene_audio + drone_scaled
            # attach scene metadata into mix_meta
            mix_meta = {**scene_meta, 'hard_negative_used': bool(hard_negative_used)}

            # optional propagation (disabled by default)
            propagation_cfg = cfg.get('propagation', {})
            if propagation_cfg.get('enabled', False):
                distance_m = category.get('distance_m') if isinstance(category, dict) else None
                mixed, att_meta = distance_attenuation_transform(
                    mixed, sr, rng, {'category': cat_name},
                    distance_m=distance_m,
                    alpha=propagation_cfg.get('alpha', 1.0),
                    ref_distance=propagation_cfg.get('ref_distance', 1.0)
                )
                mixed, lpf_meta = air_absorption_lpf_transform(
                    mixed, sr, rng, {'category': cat_name},
                    distance_m=distance_m,
                    base_fc=propagation_cfg.get('base_fc', 8000.0),
                    beta=propagation_cfg.get('beta', 0.5),
                    ref_distance=propagation_cfg.get('ref_distance', 1.0),
                    min_fc=propagation_cfg.get('min_fc', 500.0)
                )
                # optional RIR convolution
                rir_cfg = propagation_cfg.get('rir', {})
                if rir_cfg.get('enabled', False):
                    mixed, rir_meta = rir_convolution_transform(
                        mixed, sr, rng, {'category': cat_name}, rir_cfg, distance_m=distance_m
                    )
                else:
                    rir_meta = {}
            else:
                att_meta = {}
                lpf_meta = {}
                rir_meta = {}

            # hardware
            # optional propagation for no-drone samples as well
            propagation_cfg = cfg.get('propagation', {})
            if propagation_cfg.get('enabled', False):
                distance_m = category.get('distance_m') if isinstance(category, dict) else None
                mixed, att_meta = distance_attenuation_transform(
                    mixed, sr, rng, {'category': cat_name},
                    distance_m=distance_m,
                    alpha=propagation_cfg.get('alpha', 1.0),
                    ref_distance=propagation_cfg.get('ref_distance', 1.0)
                )
                mixed, lpf_meta = air_absorption_lpf_transform(
                    mixed, sr, rng, {'category': cat_name},
                    distance_m=distance_m,
                    base_fc=propagation_cfg.get('base_fc', 8000.0),
                    beta=propagation_cfg.get('beta', 0.5),
                    ref_distance=propagation_cfg.get('ref_distance', 1.0),
                    min_fc=propagation_cfg.get('min_fc', 500.0)
                )
                rir_cfg = propagation_cfg.get('rir', {})
                if rir_cfg.get('enabled', False):
                    mixed, rir_meta = rir_convolution_transform(
                        mixed, sr, rng, {'category': cat_name}, rir_cfg, distance_m=distance_m
                    )
                else:
                    rir_meta = {}
            else:
                att_meta = {}
                lpf_meta = {}
                rir_meta = {}

            hw_audio, hw_meta = hardware_transform(mixed, sr, rng, {}, cfg.get('mems_simulation', {}))

            # post
            final_audio, post_meta = post_transform(hw_audio, sr, rng, {}, cfg.get('audio_parameters', {}))

            relpath = f"1/{cat_name}_{i}.wav"
            if not dry_run:
                write_wav(out / relpath, final_audio, sr)
            seed = int(seed_from_key(master_seed, seed_key))
            # assemble extras from transform metas
            extras = {}
            for d in (meta_delta, mix_meta, att_meta, lpf_meta, rir_meta, hw_meta, post_meta):
                for k, v in (d or {}).items():
                    if k == 'noise_buffers':
                        continue
                    try:
                        import numpy as _np
                        if isinstance(v, (_np.floating, _np.integer)):
                            extras[k] = float(v)
                            continue
                    except Exception:
                        pass
                    if hasattr(v, 'tolist') and not isinstance(v, dict):
                        continue
                    extras[k] = v

            # Ensure schema consistency across classes
            # Place potentially class-conditional fields (distance, target_snr) into debug_meta
            distance_m = category.get('distance_m') if isinstance(category, dict) else None
            target_snr_val = float(category.get('snr_db', category.get('target_snr_db', None)))
            train_meta = {
                'relpath': relpath,
                'seed_key': seed_key,
                'seed': seed,
                'label': 1,
                'category': cat_name,
                'actual_snr_db_preexport': extras.get('actual_snr_db_preexport'),
                'peak_dbfs': extras.get('peak_dbfs'),
                'rms_dbfs': extras.get('rms_dbfs'),
                'clip_count': int(extras.get('clip_count', 0))
            }
            # debug_meta contains full extras and provenance including target_snr and distance
            debug_meta = {**extras, 'drone_source': str(drone_file) if drone_file is not None else None, 'mix_meta': mix_meta,
                          'distance_m': distance_m, 'target_snr_db': target_snr_val}

            out_obj = {'train_meta': train_meta, 'debug_meta': debug_meta}
            append_jsonl_line(stream, out_obj)
            samples_meta.append(out_obj)
            total_generated += 1

    # No-drone categories
    for category in cfg.get('no_drone_augmentation', {}).get('categories', []):
        cat_name = category['name']
        samples_count = max(1, int(cfg['output']['samples_per_category_no_drone'] * category.get('proportion', 1)))
        for i in range(samples_count):
            seed_key = f"{cat_name}|{i}"
            rng = rng_for_key(master_seed, seed_key)
            noise_files = rng.choice(no_drone_files, size=min(int(category.get('num_noise_sources', category.get('num_background_noises',1))), len(no_drone_files)), replace=False).tolist() if no_drone_files else []
            # If scene enabled, build scene from pool; otherwise use legacy source_transform/mix
            scene_cfg = cfg.get('scene', {})
            if scene_cfg.get('enabled', False):
                noise_pool = scene_cfg.get('noise_pool_dirs') or [str(no_drone_dir)]
                # hard negatives for no-drone
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
            relpath = f"0/{cat_name}_{i}.wav"
            if not dry_run:
                write_wav(out / relpath, final_audio, sr)
            seed = int(seed_from_key(master_seed, seed_key))
            extras = {}
            for d in (meta_delta, mix_meta, att_meta, lpf_meta, hw_meta, post_meta):
                for k, v in (d or {}).items():
                    if k == 'noise_buffers':
                        continue
                    try:
                        import numpy as _np
                        if isinstance(v, (_np.floating, _np.integer)):
                            extras[k] = float(v)
                            continue
                    except Exception:
                        pass
                    if hasattr(v, 'tolist') and not isinstance(v, dict):
                        continue
                    extras[k] = v

            # Ensure same train_meta schema as drone
            # For no-drone, sample target_snr_db and distance_m from drone pools to avoid class-conditional fields
            distance_m = category.get('distance_m') if isinstance(category, dict) else None
            sampled_target_snr = None
            sampled_distance = None
            if _drone_snr_pool:
                sampled_target_snr = float(rng.choice(_drone_snr_pool))
            if (_drone_distance_pool) and (distance_m is None):
                sampled_distance = int(rng.choice(_drone_distance_pool))

            train_meta = {
                'relpath': relpath,
                'seed_key': seed_key,
                'seed': seed,
                'label': 0,
                'category': cat_name,
                'actual_snr_db_preexport': extras.get('actual_snr_db_preexport'),
                'peak_dbfs': extras.get('peak_dbfs'),
                'rms_dbfs': extras.get('rms_dbfs'),
                'clip_count': int(extras.get('clip_count', 0))
            }
            debug_meta = {**extras, 'hard_negative_used': mix_meta.get('hard_negative_used') if isinstance(mix_meta, dict) else None,
                          'distance_m': sampled_distance if sampled_distance is not None else distance_m,
                          'target_snr_db': sampled_target_snr}
            out_obj = {'train_meta': train_meta, 'debug_meta': debug_meta}
            append_jsonl_line(stream, out_obj)
            samples_meta.append(out_obj)
            total_generated += 1

    # summary
    summary_obj = {
        'generation_time': None,
        'version': '4.0-migration-skeleton',
        'total_generated': total_generated,
        'samples': samples_meta
    }
    write_summary(Path(out) / 'augmentation_summary.json', summary_obj)

    return total_generated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    run_smoke(args.config, args.out_dir, dry_run=False)

if __name__ == '__main__':
    main()
