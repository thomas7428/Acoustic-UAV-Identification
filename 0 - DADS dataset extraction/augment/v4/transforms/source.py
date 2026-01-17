"""Legacy-like source transform for v4 local copy.

Provides `source_transform(sample_spec, sr, rng, meta)` which returns
(audio, meta_delta). This is a lightweight implementation sufficient for
smoke testing: it reads a provided `drone_file` or sums provided `noise_files`.
"""
from pathlib import Path
import numpy as np
import soundfile as sf


def _load_mono(path: Path, target_sr: int):
    x, sr = sf.read(str(path), dtype='float32', always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if sr != target_sr:
        # naive resample
        old_idx = np.linspace(0, len(x) - 1, num=len(x))
        new_len = int(round(len(x) * target_sr / sr))
        new_idx = np.linspace(0, len(x) - 1, num=new_len)
        x = np.interp(new_idx, old_idx, x).astype(np.float32)
    return x


def source_transform(sample_spec, sr, rng, meta):
    t = sample_spec.get('type')
    dur_s = float(sample_spec.get('duration_s', 1.0))
    length = int(sr * dur_s)
    if t == 'drone' and sample_spec.get('drone_file'):
        p = Path(sample_spec['drone_file'])
        if p.exists():
            x = _load_mono(p, sr)
            # trim or pad
            if len(x) > length:
                x = x[:length]
            elif len(x) < length:
                x = np.pad(x, (0, length - len(x)))
            return x.astype(np.float32), {'source': str(p.name)}
    # no drone: mix noise files if provided
    noise_files = sample_spec.get('noise_files') or []
    if noise_files:
        parts = []
        for nf in noise_files:
            p = Path(nf)
            if p.exists():
                try:
                    parts.append(_load_mono(p, sr))
                except Exception:
                    continue
        if parts:
            # sum with simple normalization
            mix = np.zeros(length, dtype=np.float32)
            for i, part in enumerate(parts):
                if len(part) < length:
                    part = np.pad(part, (0, max(0, length - len(part))))
                mix[:len(part)] += part[:length]
            mix = mix / max(1.0, float(len(parts)))
            return mix.astype(np.float32), {'num_noise_sources': len(parts)}

    # default: silence
    return np.zeros(length, dtype=np.float32), {'source': 'silence'}
