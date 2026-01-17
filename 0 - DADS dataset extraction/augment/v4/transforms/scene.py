"""Scene mixer transform: assemble acoustic scenes from noise stems.

Deterministic selection via provided `rng`. Caches short stems in memory
for performance (simple cache). Outputs scene audio of requested length
and metadata describing stems, offsets, gains and events.
"""
from pathlib import Path
import numpy as np
import soundfile as sf
import hashlib
import json
import math

# simple cache: {path: (arr, sr)}
_STEM_CACHE = {}


def _load_stem(p: Path, target_sr: int):
    key = (str(p.resolve()), int(target_sr))
    if key in _STEM_CACHE:
        return _STEM_CACHE[key]
    x, sr = sf.read(str(p), dtype='float32', always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if sr != target_sr:
        # naive linear resample
        old_idx = np.linspace(0, len(x) - 1, num=len(x))
        new_len = int(round(len(x) * target_sr / sr))
        new_idx = np.linspace(0, len(x) - 1, num=new_len)
        x = np.interp(new_idx, old_idx, x).astype(np.float32)
    _STEM_CACHE[key] = (x, target_sr)
    return _STEM_CACHE[key]


def _rms(x):
    return math.sqrt(float(np.mean(np.square(x)))) if x.size else 0.0


def _make_envelope(length, sr, rng, segments=4, min_active=0.4):
    # create intermittent envelope with smooth fades
    env = np.zeros(length, dtype=np.float32)
    seg_len = length // segments
    for s in range(segments):
        start = s * seg_len
        end = start + seg_len if s < segments - 1 else length
        active = rng.random() < min_active
        if active:
            # random activity within segment
            a0 = start + rng.integers(0, max(1, seg_len // 4))
            a1 = min(end, a0 + rng.integers(max(1, seg_len // 4), seg_len))
            win = np.hanning(a1 - a0) if a1 - a0 > 2 else np.ones(a1 - a0)
            env[a0:a1] = win
    # smooth edges
    env = np.clip(env, 0.0, 1.0)
    return env


def scene_mix_transform(audio_or_none, sr, rng, meta, scene_cfg=None, noise_pool_dirs=None):
    """
    Build a scene track from noise pool.

    Parameters:
      audio_or_none: ignored (signature compatible with other transforms)
      sr: sample rate
      rng: numpy RNG
      meta: metadata dict
      scene_cfg: dict controlling stems range, envelopes, events
      noise_pool_dirs: list of folders to sample stems from

    Returns: (scene_audio (1-d np.float32), meta_delta)
    """
    if scene_cfg is None:
        scene_cfg = {}
    if noise_pool_dirs is None:
        noise_pool_dirs = []
    # collect stems
    stems = []
    for d in noise_pool_dirs:
        p = Path(d)
        if p.exists():
            stems.extend(sorted([str(x) for x in p.glob('*.wav')]))
    if not stems:
        return np.zeros(int(sr * float(scene_cfg.get('duration_s', 1.0))), dtype=np.float32), {'scene_id': None, 'num_stems': 0}

    min_k = int(scene_cfg.get('min_stems', 1))
    max_k = int(scene_cfg.get('max_stems', 4))
    K = int(rng.integers(min_k, max_k + 1))
    K = min(K, len(stems))

    chosen = rng.choice(stems, size=K, replace=False).tolist()

    dur_s = float(scene_cfg.get('duration_s', 1.0))
    length = int(sr * dur_s)
    scene = np.zeros(length, dtype=np.float32)

    offsets_s = []
    gains_db = []
    envelope_params = None

    for s_path in chosen:
        arr, _ = _load_stem(Path(s_path), sr)
        # choose random offset so that stem overlaps scene
        if len(arr) > length:
            max_start = len(arr) - length
            st = int(rng.integers(0, max_start + 1))
            seg = arr[st:st + length]
            offset = float(st / sr)
        else:
            # loop or pad
            rep = int(np.ceil(length / max(1, len(arr))))
            seg = np.tile(arr, rep)[:length]
            offset = 0.0
        # per-stem gain
        gain_db = float(rng.uniform(scene_cfg.get('gain_db_min', -6.0), scene_cfg.get('gain_db_max', 0.0)))
        gain = 10.0 ** (gain_db / 20.0)
        # envelope
        env = _make_envelope(length, sr, rng, segments=int(scene_cfg.get('env_segments', 4)), min_active=float(scene_cfg.get('env_activity_prob', 0.6)))
        seg = seg * env * gain
        scene += seg
        offsets_s.append(offset)
        gains_db.append(gain_db)
        if envelope_params is None:
            envelope_params = {'segments': int(scene_cfg.get('env_segments', 4)), 'activity_prob': float(scene_cfg.get('env_activity_prob', 0.6))}

    # optional transient events
    event_count = 0
    if scene_cfg.get('events', {}).get('enabled', False):
        ev_cfg = scene_cfg.get('events', {})
        max_events = int(ev_cfg.get('max_per_scene', 3))
        event_count = int(rng.integers(0, max_events + 1))
        for _ in range(event_count):
            pos = int(rng.integers(0, length))
            ev_dur = int(max(1, int(ev_cfg.get('duration_frames', int(0.02 * sr)))))
            burst = rng.normal(scale=0.5, size=min(ev_dur, length - pos)).astype(np.float32)
            scene[pos:pos + len(burst)] += burst

    # scene id: deterministic hash of chosen stems + offsets + gains
    h = hashlib.sha256()
    h.update(json.dumps({'stems': chosen, 'offsets': offsets_s, 'gains': gains_db}, sort_keys=True).encode('utf-8'))
    scene_id = h.hexdigest()[:16]

    meta_delta = {
        'scene_id': scene_id,
        'num_stems': int(K),
        'stems': chosen,
        'offsets_s': [float(x) for x in offsets_s],
        'gains_db': [float(x) for x in gains_db],
        'envelope_params': envelope_params,
        'event_count': int(event_count)
    }

    # ensure no clipping here (post will handle anti-clip)
    return scene.astype(np.float32), meta_delta
