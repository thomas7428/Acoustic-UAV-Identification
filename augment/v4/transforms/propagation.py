from pathlib import Path
import numpy as np

def _infer_distance_from_category(category_name: str):
    # parse trailing pattern like *_<digits>m, e.g. drone_1000m
    try:
        if category_name and category_name.endswith('m'):
            num = ''
            for c in reversed(category_name[:-1]):
                if c.isdigit():
                    num = c + num
                else:
                    break
            if num:
                return float(num)
    except Exception:
        pass
    return 1.0

def distance_attenuation_transform(audio, sr, rng, meta, distance_m=None, alpha=1.0, ref_distance=1.0):
    """Apply simple inverse-power distance attenuation: gain = (ref / max(d,ref))**alpha

    Returns (audio, meta_delta)
    """
    if distance_m is None:
        distance_m = meta.get('category') and _infer_distance_from_category(meta.get('category')) or ref_distance
    d = max(distance_m, ref_distance)
    gain = (ref_distance / d) ** float(alpha)
    gain_db = 20.0 * np.log10(gain) if gain > 0 else -999.0
    out = audio * float(gain)
    meta_delta = {'distance_m': float(distance_m), 'alpha': float(alpha), 'gain_db': float(gain_db)}
    return out, meta_delta

def air_absorption_lpf_transform(audio, sr, rng, meta, distance_m=None, base_fc=8000.0, beta=0.5, ref_distance=1.0, min_fc=500.0):
    """Apply a distance-dependent first-order lowpass filter approximating air absorption.

    fc decreases with distance via: fc = max(min_fc, base_fc * (ref_distance / max(distance, ref_distance))**beta)
    Uses a 1-pole IIR lowpass: y[n] = a*x[n] + (1-a)*y[n-1] with a = 1 - exp(-2*pi*fc/sr)
    """
    if distance_m is None:
        distance_m = meta.get('category') and _infer_distance_from_category(meta.get('category')) or ref_distance
    d = max(distance_m, ref_distance)
    fc = max(min_fc, float(base_fc) * (ref_distance / d) ** float(beta))

    # compute coefficient
    a = 1.0 - np.exp(-2.0 * np.pi * fc / float(sr))
    # apply single-pole lowpass
    y = np.empty_like(audio)
    if audio.size == 0:
        return audio, {'distance_m': float(distance_m), 'fc_hz': float(fc)}
    y[0] = a * audio[0]
    for n in range(1, len(audio)):
        y[n] = a * audio[n] + (1.0 - a) * y[n-1]

    meta_delta = {'distance_m': float(distance_m), 'fc_hz': float(fc)}
    return y, meta_delta
