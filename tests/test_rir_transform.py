import numpy as np
from augment.v4.transforms.rir import rir_convolution_transform, _load_ir_bank


def test_rir_deterministic_and_length():
    sr = 22050
    rng = np.random.default_rng(42)
    # make tiny IR bank
    import tempfile, soundfile as sf, os
    td = tempfile.TemporaryDirectory()
    for i in range(3):
        x = (np.random.default_rng(i+1).normal(size=int(0.1 * sr)) * np.exp(-np.linspace(0,1,int(0.1*sr)))).astype('float32')
        p = os.path.join(td.name, f'rir_{i}.wav')
        sf.write(p, x, sr)

    bank = _load_ir_bank(td.name, sr)
    assert len(bank) >= 1

    audio = np.zeros(int(0.5 * sr), dtype=np.float32)
    audio[100:200] = 1.0
    out1, meta1 = rir_convolution_transform(audio, sr, rng, {}, {'enabled': True, 'rir_dir': td.name, 'dry_wet': [0.3,0.3]}, distance_m=100)
    rng2 = np.random.default_rng(42)
    out2, meta2 = rir_convolution_transform(audio, sr, rng2, {}, {'enabled': True, 'rir_dir': td.name, 'dry_wet': [0.3,0.3]}, distance_m=100)
    assert out1.shape == audio.shape
    assert not np.any(np.isnan(out1))
    # deterministic
    assert np.allclose(out1, out2)
    assert 'rir_id' in meta1 and 'dry_wet' in meta1
