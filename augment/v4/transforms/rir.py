"""RIR convolution transform for v4 pipeline.

Deterministic selection from an IR bank, FFT convolution, crop/pad to original
length, dry/wet mixing, and per-sample metadata emission.
"""
from pathlib import Path
import numpy as np
import soundfile as sf
import math

# module-level cache for loaded IRs: {rir_dir: [(id, arr, sr), ...]}
_IR_CACHE = {}


def _load_ir_bank(rir_dir: str, target_sr: int):
    p = Path(rir_dir)
    key = (str(p.resolve()), int(target_sr))
    if key in _IR_CACHE:
        return _IR_CACHE[key]
    items = []
    if not p.exists():
        _IR_CACHE[key] = items
        return items
    for f in sorted(p.glob('*.wav')):
        try:
            x, sr = sf.read(f, dtype='float32', always_2d=False)
            x = np.asarray(x, dtype=np.float32)
            if sr != target_sr:
                # simple resample using linear interpolation if necessary
                import warnings
                warnings.warn('RIR sample rate mismatch; resampling')
                # naive resample
                old_idx = np.linspace(0, len(x) - 1, num=len(x))
                new_len = int(round(len(x) * target_sr / sr))
                new_idx = np.linspace(0, len(x) - 1, num=new_len)
                x = np.interp(new_idx, old_idx, x).astype(np.float32)
            items.append((f.name, x, target_sr))
        except Exception:
            continue
    _IR_CACHE[key] = items
    return items


def _rms_db(x: np.ndarray):
    eps = 1e-12
    rms = math.sqrt(float(np.mean(np.square(x)))) if x.size else 0.0
    return 20.0 * math.log10(rms + eps)


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def _fft_convolve(x: np.ndarray, h: np.ndarray):
    # rfft-based linear convolution
    n = len(x) + len(h) - 1
    N = _next_pow2(n)
    X = np.fft.rfft(x, n=N)
    H = np.fft.rfft(h, n=N)
    y = np.fft.irfft(X * H, n=N)
    return y[:n]


def rir_convolution_transform(audio, sr, rng, meta, rir_cfg=None, distance_m=None):
    """
    audio: 1-d numpy float array
    sr: sample rate
    rng: numpy.random.Generator
    meta: existing metadata dict
    rir_cfg: dict with keys described in task

    returns (audio_out, meta_delta)
    """
    if rir_cfg is None:
        rir_cfg = {}
    enabled = bool(rir_cfg.get('enabled', False))
    if not enabled:
        return audio, {'rir_enabled': False}

    rir_dir = rir_cfg.get('rir_dir')
    strategy = rir_cfg.get('select', {}).get('strategy', 'rng')
    dry_wet_min, dry_wet_max = rir_cfg.get('dry_wet', [0.2, 0.6])
    normalize_rir = bool(rir_cfg.get('normalize_rir', True))

    bank = _load_ir_bank(rir_dir or '', sr)
    if not bank:
        # nothing to do
        return audio, {'rir_enabled': False, 'rir_note': 'no_ir_found'}

    # choose IR deterministically
    if strategy == 'by_distance' and distance_m is not None:
        # map distance to index by proportional rank
        # assume distances in meta.category when present (e.g., 'drone_300m')
        # simple proportional mapping across bank
        # clamp distance to reasonable bounds
        d = float(distance_m)
        # for mapping we'll use an arbitrary cap of 2000m
        frac = min(max(d / 2000.0, 0.0), 1.0)
        idx = int(round(frac * (len(bank) - 1)))
    else:
        idx = int(rng.integers(0, len(bank)))

    rir_id, h, h_sr = bank[idx]
    # optional normalize RIR to unity RMS
    if normalize_rir:
        rms = np.sqrt(float(np.mean(np.square(h)))) if h.size else 1.0
        if rms > 0:
            h = h / rms

    # convolve via FFT
    conv = _fft_convolve(audio, h)
    # crop/pad to original length
    out_len = len(audio)
    conv_cropped = conv[:out_len]
    if len(conv_cropped) < out_len:
        conv_cropped = np.pad(conv_cropped, (0, out_len - len(conv_cropped)))

    # dry/wet
    dry_wet = float(rng.uniform(dry_wet_min, dry_wet_max))
    out = (1.0 - dry_wet) * audio + dry_wet * conv_cropped

    meta_delta = {
        'rir_enabled': True,
        'rir_mode': strategy,
        'rir_id': rir_id,
        'dry_wet': float(dry_wet),
        'rir_len': int(len(h)),
        'rir_energy_db': float(_rms_db(h)),
        'conv_method': 'fft'
    }
    return out.astype(np.float32), meta_delta
