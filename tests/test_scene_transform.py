import numpy as np
from augment.v4.transforms.scene import scene_mix_transform


def test_scene_deterministic_and_length():
    sr = 22050
    rng = np.random.default_rng(12345)
    cfg = {'min_stems': 1, 'max_stems': 2, 'duration_s': 0.5, 'env_segments': 2, 'env_activity_prob': 0.8}
    pool = []
    # create two tiny synthetic stems in memory by writing to tmp files
    import tempfile, soundfile as sf
    td = tempfile.TemporaryDirectory()
    for i in range(2):
        x = (np.random.default_rng(i).normal(scale=0.2, size=int(0.4 * sr))).astype('float32')
        p = f"{td.name}/stem_{i}.wav"
        sf.write(p, x, sr)
        pool.append(p)

    a1, m1 = scene_mix_transform(None, sr, rng, {}, cfg, pool)
    rng2 = np.random.default_rng(12345)
    a2, m2 = scene_mix_transform(None, sr, rng2, {}, cfg, pool)
    assert a1.shape == a2.shape
    assert not np.any(np.isnan(a1))
    assert np.allclose(a1, a2)
    assert 'scene_id' in m1 and 'num_stems' in m1
