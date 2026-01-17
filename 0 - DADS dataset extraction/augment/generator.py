from pathlib import Path
import json
import sys
from datetime import datetime, timezone
import hashlib

from .io import atomic_write_json
# reuse existing audio utilities from project for loading/resampling
from pathlib import Path as _Path
import sys as _sys
# ensure workspace root on sys.path so `tools` can be imported
_root = _Path(__file__).resolve().parents[2]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
from tools.audio_utils import load_audio_file
from .validation import validate_build_config
from .writer import SplitWriter
from .rng import rng_for_key
from . import pipeline
from .distance import apply as distance_apply
from .modifiers import normalize as normalize_mod
import os
import multiprocessing as mp
from typing import Dict, Any


def process_task(task: Dict[str, Any]):
    """Top-level worker function: load offline source (if any), run pipeline, return processed audio and metadata."""
    try:
        from .rng import rng_for_key as _rng_for_key
        from tools.audio_utils import load_audio_file as _load_audio_file
        from .modifiers import normalize as _normalize_mod
        from .modifiers import environment as _env_mod
        from .distance import apply as _distance_apply
        from . import pipeline as _pipeline
        import numpy as _np

        rng = _rng_for_key(task['master_seed'], task['seed_key'])

        cfg = task.get('cfg', {})
        dist_cfg = cfg.get('distance', {}) if isinstance(cfg, dict) else {}
        ref = float(dist_cfg.get('ref_m', 1.0))
        alpha = float(dist_cfg.get('alpha', 1.8))

        def _atten_linear_from_dist(d):
            import math
            att_dB = -20.0 * alpha * math.log10(max(d / ref, 1e-6))
            return float(10 ** (att_dB / 20.0)), float(att_dB)

        # collect scenario distances from task
        scenario = task.get('scenario', {})
        drones_ds = scenario.get('drones', []) if isinstance(scenario, dict) else []
        ambients_ds = scenario.get('ambients', []) if isinstance(scenario, dict) else []

        # pools
        pool_drone = list(task.get('offline_pool_label_1', []))
        pool_ambient = list(task.get('offline_pool_label_0', []))

        # load and assemble drone mix
        drone_parts = []
        drone_debug = []
        for idx, ddist in enumerate(drones_ds):
            if not pool_drone:
                break
            try:
                src = rng.choice(pool_drone)
                src_audio = _load_audio_file(src, task['sr'], duration=task['duration'])
                if src_audio is None:
                    continue
                # normalize source
                src_audio, nd = _normalize_mod.apply(src_audio, task['sr'], rng, task['train_meta'], task['cfg'])
                # apply pre-source reverb if requested in scenario
                rdbg = None
                try:
                    rev_cfg = None
                    if isinstance(scenario, dict) and scenario.get('reverb') is not None:
                        rev_cfg = scenario.get('reverb')
                    elif isinstance(task.get('cfg', {}), dict) and task['cfg'].get('reverb') is not None:
                        rev_cfg = task['cfg'].get('reverb')
                    if rev_cfg is not None and rev_cfg.get('mode') == 'pre':
                        # deterministic per-source reverb RNG
                        src_rng = _rng_for_key(task['master_seed'], task['seed_key'] + f':reverb:drone:{idx}')
                        meta_rev = dict(task.get('train_meta', {}))
                        meta_rev['reverb'] = rev_cfg
                        src_audio, rdbg = _reverb_mod.apply(src_audio, task['sr'], src_rng, meta_rev, task['cfg'])
                except Exception:
                    rdbg = None
                # apply environment wrapper (movement, appearance, chained modifiers)
                envdbg = None
                try:
                    env_cfg = None
                    if isinstance(scenario, dict) and scenario.get('environment') is not None:
                        env_cfg = scenario.get('environment')
                    elif isinstance(task.get('cfg', {}), dict) and task['cfg'].get('environment') is not None:
                        env_cfg = task['cfg'].get('environment')
                    if env_cfg is not None:
                        src_rng_env = _rng_for_key(task['master_seed'], task['seed_key'] + f':env:drone:{idx}')
                        meta_env = dict(task.get('train_meta', {}))
                        env_audio, envdbg = _env_mod.apply(src_audio, task['sr'], src_rng_env, meta_env, env_cfg)
                        # replace audio only if returned
                        if env_audio is not None:
                            src_audio = env_audio
                except Exception:
                    envdbg = None
                gain, att_db = _atten_linear_from_dist(float(ddist))
                drone_parts.append((src_audio, gain))
                # store only filename for readability in JSONL
                try:
                    from pathlib import Path as _P
                    src_name = _P(src).name
                except Exception:
                    src_name = src
                entry = {'src': src_name, 'distance_m': float(ddist), 'atten_dB': att_db}
                if rdbg is not None:
                    entry['reverb_debug'] = rdbg
                if envdbg is not None:
                    entry['env_debug'] = envdbg
                drone_debug.append(entry)
            except Exception:
                continue

        import numpy as _np
        def _mix_parts(parts):
            if not parts:
                return None
            lengths = [p[0].size for p in parts if p[0] is not None]
            if not lengths:
                return None
            L = max(lengths)
            out = _np.zeros(L, dtype=float)
            for a, g in parts:
                aa = a.astype(float)
                if aa.size < L:
                    aa = _np.pad(aa, (0, L - aa.size))
                out += aa * float(g)
            return out

        base_audio = _mix_parts(drone_parts)

        # prepare ambient mix_sources list
        mix_sources = []
        ambient_debug = []
        # If scenario specifies snr_db, we will ignore ambient distances and let mix.apply
        # scale ambient gains to meet the SNR relative to the strongest drone.
        scenario_snr = None
        if isinstance(scenario, dict) and 'snr_db' in scenario:
            try:
                scenario_snr = float(scenario.get('snr_db'))
            except Exception:
                scenario_snr = None

        for idx, adist in enumerate(ambients_ds):
            if not pool_ambient:
                break
            try:
                src = rng.choice(pool_ambient)
                src_audio = _load_audio_file(src, task['sr'], duration=task['duration'])
                if src_audio is None:
                    continue
                src_audio, nd = _normalize_mod.apply(src_audio, task['sr'], rng, task['train_meta'], task['cfg'])
                # apply pre-source reverb if requested in scenario
                rdbg = None
                try:
                    rev_cfg = None
                    if isinstance(scenario, dict) and scenario.get('reverb') is not None:
                        rev_cfg = scenario.get('reverb')
                    elif isinstance(task.get('cfg', {}), dict) and task['cfg'].get('reverb') is not None:
                        rev_cfg = task['cfg'].get('reverb')
                    if rev_cfg is not None and rev_cfg.get('mode') == 'pre':
                        src_rng = _rng_for_key(task['master_seed'], task['seed_key'] + f':reverb:ambient:{idx}')
                        meta_rev = dict(task.get('train_meta', {}))
                        meta_rev['reverb'] = rev_cfg
                        src_audio, rdbg = _reverb_mod.apply(src_audio, task['sr'], src_rng, meta_rev, task['cfg'])
                except Exception:
                    rdbg = None
                # apply environment wrapper (movement, appearance, chained modifiers)
                envdbg = None
                try:
                    env_cfg = None
                    if isinstance(scenario, dict) and scenario.get('environment') is not None:
                        env_cfg = scenario.get('environment')
                    elif isinstance(task.get('cfg', {}), dict) and task['cfg'].get('environment') is not None:
                        env_cfg = task['cfg'].get('environment')
                    if env_cfg is not None:
                        src_rng_env = _rng_for_key(task['master_seed'], task['seed_key'] + f':env:ambient:{idx}')
                        meta_env = dict(task.get('train_meta', {}))
                        env_audio, envdbg = _env_mod.apply(src_audio, task['sr'], src_rng_env, meta_env, env_cfg)
                        if env_audio is not None:
                            src_audio = env_audio
                except Exception:
                    envdbg = None

                if scenario_snr is not None:
                    # initial gain placeholder; actual scaling will be done by mix.apply
                    gain = 1.0
                    try:
                        from pathlib import Path as _P
                        src_name = _P(src).name
                    except Exception:
                        src_name = src
                    entry = {'src': src_name, 'distance_m': float(adist), 'atten_dB': None, 'note': 'distance ignored due to snr_db'}
                    if rdbg is not None:
                        entry['reverb_debug'] = rdbg
                    if envdbg is not None:
                        entry['env_debug'] = envdbg
                    ambient_debug.append(entry)
                else:
                    gain, att_db = _atten_linear_from_dist(float(adist))
                    try:
                        from pathlib import Path as _P
                        src_name = _P(src).name
                    except Exception:
                        src_name = src
                    entry = {'src': src_name, 'distance_m': float(adist), 'atten_dB': att_db}
                    if rdbg is not None:
                        entry['reverb_debug'] = rdbg
                    if envdbg is not None:
                        entry['env_debug'] = envdbg
                    ambient_debug.append(entry)
                mix_sources.append({'audio': src_audio, 'gain': float(gain), 'distance_m': (float(adist) if adist is not None else None), 'src_name': src_name})
            except Exception:
                continue

        # if no drone parts and we have a single offline_src (legacy), try to load it
        if base_audio is None and task.get('offline_src'):
            try:
                base_audio = _load_audio_file(task['offline_src'], task['sr'], duration=task['duration'])
                if base_audio is not None:
                    base_audio, nd = _normalize_mod.apply(base_audio, task['sr'], rng, task['train_meta'], task['cfg'])
            except Exception:
                base_audio = None

        # assemble meta for mix.apply
        mix_meta = dict(task.get('train_meta', {}))
        mix_meta['mix_sources'] = mix_sources

        # call pure-array mixer
        mix_debug = {}
        try:
            from .modifiers import mix as _mix_mod
            from .modifiers import reverb as _reverb_mod
            if scenario_snr is not None and base_audio is not None and drone_parts:
                # find the strongest drone (max rms * gain)
                strongest = None
                strongest_power = -1.0
                for (a, g), dbg in zip(drone_parts, drone_debug):
                    r = float(_mix_mod._rms(a)) * float(g) if a is not None else 0.0
                    if r > strongest_power:
                        strongest_power = r
                        strongest = (a, g)
                ref_audio = strongest[0] if strongest is not None else base_audio
                # ask mix.apply to scale ambients to meet target SNR relative to ref_audio
                mix_meta['target_snr_db'] = float(scenario_snr)
                processed_ref_sum, mix_debug = _mix_mod.apply(ref_audio, task['sr'], rng, mix_meta, task['cfg'])
                # extract scaled ambient (processed_ref_sum - ref_audio)
                import numpy as _np
                def _ensure_len_local(arr, L):
                    if arr is None:
                        return _np.zeros(L, dtype=float)
                    a = arr.astype(float)
                    if a.size == L:
                        return a
                    if a.size > L:
                        return a[:L]
                    return _np.pad(a, (0, L - a.size))
                L = 0
                if ref_audio is not None:
                    L = max(L, int(ref_audio.size))
                if processed_ref_sum is not None:
                    L = max(L, int(processed_ref_sum.size))
                if L == 0:
                    ambient_scaled = _np.zeros(int(task['sr'] * task['duration']), dtype=float)
                else:
                    ref = _ensure_len_local(ref_audio, L)
                    scaled_total = _ensure_len_local(processed_ref_sum, L)
                    ambient_scaled = scaled_total - ref
                # now add ambient_scaled to full drone mix (base_audio)
                if base_audio is None:
                    processed_audio = ambient_scaled
                else:
                    ba = base_audio.astype(float) if base_audio is not None else _np.zeros(0, dtype=float)
                    L2 = max(ba.size, ambient_scaled.size)
                    if ba.size < L2:
                        ba = _np.pad(ba, (0, L2 - ba.size))
                    if ambient_scaled.size < L2:
                        ambient_scaled = _np.pad(ambient_scaled, (0, L2 - ambient_scaled.size))
                    processed_audio = ba + ambient_scaled
            else:
                processed_audio, mix_debug = _mix_mod.apply(base_audio, task['sr'], rng, mix_meta, task['cfg'])
        except Exception as e:
            # fallback: naive sum
            import numpy as _np
            processed_audio = base_audio.copy() if base_audio is not None else _np.zeros(int(task['sr'] * task['duration']), dtype=float)
            for m in mix_sources:
                a = m.get('audio')
                g = float(m.get('gain', 1.0))
                if a is None:
                    continue
                aa = a.astype(float)
                if aa.size < processed_audio.size:
                    aa = _np.pad(aa, (0, processed_audio.size - aa.size))
                processed_audio += aa * g
            mix_debug = {'fallback_mix_error': str(e)}

        # final normalize
        try:
            processed_audio, nd = _normalize_mod.apply(processed_audio, task['sr'], rng, task['train_meta'], task['cfg'])
            # apply post-mix reverb if requested by scenario
            try:
                rev_cfg = None
                if isinstance(scenario, dict) and scenario.get('reverb') is not None:
                    rev_cfg = scenario.get('reverb')
                elif isinstance(task.get('cfg', {}), dict) and task['cfg'].get('reverb') is not None:
                    rev_cfg = task['cfg'].get('reverb')
                if rev_cfg is not None and rev_cfg.get('mode', 'post') == 'post':
                    mix_meta_rev = dict(task.get('train_meta', {}))
                    mix_meta_rev['reverb'] = rev_cfg
                    processed_audio, rev_debug = _reverb_mod.apply(processed_audio, task['sr'], rng, mix_meta_rev, task['cfg'])
                    # merge reverb debug into mix_debug
                    if isinstance(mix_debug, dict):
                        mix_debug['reverb_debug'] = rev_debug
                    else:
                        mix_debug = {'reverb_debug': rev_debug}
            except Exception:
                pass
            # collate debug
            debug = {'drone_parts': drone_debug, 'ambient_parts': ambient_debug}
            debug.update(mix_debug if isinstance(mix_debug, dict) else {})
        except Exception as e:
            debug = {'error': str(e)}

        return {
            'filename': task['filename'],
            'split': task['split'],
            'label': task['label'],
            'processed_audio': processed_audio,
            'train_meta': task['train_meta'],
            'debug': debug,
            'offline_source': task.get('offline_src'),
        }
    except Exception as e:
        return {'error': str(e), 'task': task}


def _sha256(path: Path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open('rb') as fh:
        while True:
            b = fh.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run_build(build_config_path: str, out_dir: str, dry_run=False, show_progress=False, total=None, seed=None):
    cfg_path = Path(build_config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    validate_build_config(cfg_path, cfg)

    info_name = cfg.get('output', {}).get('info_filename', 'augmentation_samples.jsonl')

    # prepare split dirs and meta files
    splits = ('dataset_train', 'dataset_val', 'dataset_test')
    for s in splits:
        d = out / s
        d.mkdir(parents=True, exist_ok=True)
        meta = d / info_name
        # Always truncate the meta file so runs are idempotent and overwrite previous
        meta.write_text('', encoding='utf-8')

    # determine totals (use total_samples and split_ratio)
    total_samples = int(cfg.get('output', {}).get('total_samples', 20))
    split_ratio = cfg.get('output', {}).get('split_ratio', [0.8, 0.1, 0.1])

    def _counts(total):
        train = int(total * split_ratio[0])
        val = int(total * split_ratio[1])
        test = int(total * split_ratio[2])
        rem = total - (train + val + test)
        train += rem
        return (train, val, test)

    per_split_counts = _counts(total_samples)

    master_seed = int(cfg.get('advanced', {}).get('random_seed', 0))
    sr = int(cfg.get('audio_parameters', {}).get('sample_rate', 22050))
    duration = float(cfg.get('audio_parameters', {}).get('target_duration_sec', cfg.get('audio_parameters', {}).get('duration_s', 4.0)))
    class_props = cfg.get('class_proportions', {'drone': 0.5, 'no_drone': 0.5})

    # Build task list for workers; main process will perform atomic writes and JSONL appends
    global_idx = 0
    tasks = []
    offline_dir = out / 'dataset_DADS_offline'
    offline_files_all = []
    offline_pool_label = {0: [], 1: []}
    if offline_dir.exists():
        for ext in ('*.wav', '*.flac', '*.ogg'):
            offline_files_all.extend(list(offline_dir.rglob(ext)))
        # collect per-label pools (strings)
        for lbl in (0, 1):
            lbl_dir = offline_dir / str(lbl)
            if lbl_dir.exists():
                for ext in ('*.wav', '*.flac', '*.ogg'):
                    offline_pool_label[lbl].extend([str(p) for p in lbl_dir.rglob(ext)])
            else:
                # fallback: use all files filtered by filename containing label prefix
                offline_pool_label[lbl].extend([str(p) for p in offline_files_all if f"_{lbl}_" in p.name])

    # Build tasks by total samples and split counts; scenarios chosen per-sample
    idx_map = {s: i for i, s in enumerate(splits)}
    global_idx = 0
    for split_name, n in zip(splits, per_split_counts):
        for i in range(n):
            seed_key = f"sample:{global_idx}"
            filename = f"dads_sample_{global_idx:06d}.wav"

            # choose class deterministically according to class_proportions
            class_label = 0
            class_name = 'no_drone'
            try:
                props = class_props if isinstance(class_props, dict) else {'drone': 0.5, 'no_drone': 0.5}
                pd = float(props.get('drone', 0.5))
                pn = float(props.get('no_drone', 1.0 - pd))
                rng_cls = rng_for_key(master_seed, seed_key + ':class')
                r = float(rng_cls.random())
                total = pd + pn
                p_drone = pd / total if total > 0 else 0.5
                if r <= p_drone:
                    class_label = 1
                    class_name = 'drone'
                else:
                    class_label = 0
                    class_name = 'no_drone'
            except Exception:
                class_label = 0
                class_name = 'no_drone'

            # choose a scenario among those matching the class (scenarios may be drone or ambient)
            scenarios = cfg.get('augmentation_scenarios', []) if isinstance(cfg, dict) else []
            scenario = None
            if scenarios:
                try:
                    # filter scenarios by whether they contain drones
                    cand = []
                    for s in scenarios:
                        has_drone = bool(s.get('drones')) if isinstance(s, dict) else False
                        if (class_label == 1 and has_drone) or (class_label == 0 and not has_drone):
                            cand.append(s)
                    if not cand:
                        cand = scenarios
                    rng_scn = rng_for_key(master_seed, seed_key + ':scenario')
                    props = [float(s.get('proportion', 0.0)) for s in cand]
                    totalp = sum(props)
                    if totalp <= 0:
                        idx = int(rng_scn.integers(0, len(cand)))
                    else:
                        probs = [p / totalp for p in props]
                        r = float(rng_scn.random())
                        cum = 0.0
                        idx = 0
                        for j, p in enumerate(probs):
                            cum += p
                            if r <= cum:
                                idx = j
                                break
                    scenario = cand[idx]
                except Exception:
                    scenario = None

            seed_key_full = f"{class_name}:{split_name}:{i}"
            train_meta = {
                'filename': str(Path(str(class_label)) / filename),
                'label': class_label,
                'class_name': class_name,
                'seed_key': seed_key_full,
                'split': split_name,
            }

            # deterministic primary offline source candidate for legacy cases
            src = None
            pool_label = offline_pool_label.get(class_label, [])
            if pool_label:
                try:
                    rng_pick = rng_for_key(master_seed, seed_key_full + ':pick')
                    src = str(rng_pick.choice(pool_label))
                except Exception:
                    src = None

            tasks.append({
                'master_seed': master_seed,
                'seed_key': seed_key_full,
                'label': class_label,
                'class_name': class_name,
                'filename': filename,
                'split': split_name,
                'offline_src': src,
                'offline_pool_label_0': offline_pool_label[0],
                'offline_pool_label_1': offline_pool_label[1],
                'scenario': scenario,
                'train_meta': train_meta,
                'sr': sr,
                'duration': duration,
                'cfg': cfg,
            })
            global_idx += 1

    # prepare writers for main process
    writers = {s: SplitWriter(out / s, meta_name=info_name) for s in splits}

    # worker function runs as top-level `process_task` so it's pickleable by multiprocessing

    # choose number of workers
    num_workers = int(cfg.get('advanced', {}).get('num_workers', max(1, (os.cpu_count() or 1))))
    if num_workers <= 1:
        # sequential
        results = [process_task(t) for t in tasks]
    else:
        with mp.Pool(processes=num_workers) as pool:
            results = list(pool.imap_unordered(process_task, tasks))

    # main process writes outputs and metadata
    written = 0
    for r in results:
        if r is None:
            continue
        if 'error' in r:
            # skip or record error in build_info
            continue
        split = r['split']
        writer = writers.get(split)
        p = writer.write_wav(str(r['label']), r['filename'], r['processed_audio'], sr)
        debug_meta = {'master_seed': master_seed}
        debug_meta.update(r.get('debug', {}) if isinstance(r.get('debug', {}), dict) else {})
        if r.get('offline_source'):
            try:
                from pathlib import Path as _P
                debug_meta['offline_source'] = _P(r.get('offline_source')).name
            except Exception:
                debug_meta['offline_source'] = r.get('offline_source')
        writer.append_jsonl(r['train_meta'], debug_meta)
        written += 1

    build_info = {
        'dataset_id': out.name,
        'generator': 'augment.generator',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'command': ' '.join(sys.argv),
        'git_sha': None,
        'sha256_build_config': _sha256(cfg_path),
        'return_code': 0,
        'totals_requested': {'total_samples': total_samples},
        'generated': {'total_files': global_idx}
    }

    atomic_write_json(out / 'build_info.json', build_info)
    return 0
