import numpy as np
from augment.v4.transforms.propagation import air_absorption_lpf_transform, distance_attenuation_transform


def band_energy_ratio(signal, sr, lo_hz, hi_hz):
    # compute FFT power and energy ratio in band
    N = len(signal)
    if N == 0:
        return 0.0
    X = np.fft.rfft(signal)
    ps = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    total = ps[(freqs >= 0) & (freqs <= hi_hz)].sum() if (freqs <= hi_hz).any() else ps.sum()
    if total == 0:
        return 0.0
    return ps[band].sum() / total


def test_hf_rolloff_monotonic():
    sr = 16000
    # white noise fixed seed
    rng = np.random.default_rng(123)
    x = rng.standard_normal(16000 * 1).astype(np.float32)

    # apply LPF at increasing distances via air_absorption_lpf_transform
    ratios = []
    for d in [1.0, 10.0, 50.0, 200.0]:
        y, meta = air_absorption_lpf_transform(x, sr, None, {'category': 'drone_{}m'.format(int(d))}, distance_m=d, base_fc=8000.0, beta=0.5, ref_distance=1.0, min_fc=500.0)
        r = band_energy_ratio(y, sr, 3000.0, 8000.0)
        ratios.append(r)

    # expect monotonic non-increasing
    assert all(ratios[i] >= ratios[i+1] - 1e-8 for i in range(len(ratios)-1)), f"HF ratios not monotonic: {ratios}"


def test_distance_attenuation_gain():
    sr = 16000
    rng = np.random.default_rng(1)
    x = rng.standard_normal(16000).astype(np.float32)
    y1, m1 = distance_attenuation_transform(x, sr, None, {'category': 'drone_1m'}, distance_m=1.0, alpha=1.0, ref_distance=1.0)
    y10, m10 = distance_attenuation_transform(x, sr, None, {'category': 'drone_10m'}, distance_m=10.0, alpha=1.0, ref_distance=1.0)
    # y10 should have lower RMS
    rms1 = np.sqrt(np.mean(y1**2))
    rms10 = np.sqrt(np.mean(y10**2))
    assert rms10 < rms1

 