from pathlib import Path
import importlib.util
import numpy as np

# load v3 mixing helpers
v3_path = Path(__file__).resolve().parents[3] / "0 - DADS dataset extraction" / "augment_dataset_v3.py"
spec = importlib.util.spec_from_file_location("augment_v3", str(v3_path))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

def mix_transform(audio, sr, rng, meta, category_config, config):
    # audio is the drone signal or noise mixture from source_transform
    # For drone class we expect meta to contain 'noise_signals' or noise files handled earlier
    # Here we call v3.mix_drone_with_noise for identical behavior
    if meta.get('class') == 'drone':
        # expect meta to contain noise_signals in 'noise_buffers' (if assembled); else load via v3 when needed
        noise_buffers = meta.get('noise_buffers')
        target_snr = category_config.get('snr_db')
        mixed, actual_snr, mix_meta = v3.mix_drone_with_noise(audio, noise_buffers, target_snr, config, sr=sr, rng=rng)
        # merge mix_meta
        meta_delta = mix_meta
        return mixed, meta_delta
    else:
        # for no-drone, use mix_background_noises
        noise_buffers = [audio]
        mixed, nm = v3.mix_background_noises(noise_buffers, category_config.get('amplitude_range', [0.5,1.0]), config, category_config, sr, rng=rng)
        return mixed, nm
