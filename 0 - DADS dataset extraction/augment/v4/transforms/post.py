from pathlib import Path
import importlib.util
import numpy as np

# load v3 normalization/anti-clip logic if present
v3 = None
v3_path = Path(__file__).resolve().parents[3] / "0 - DADS dataset extraction" / "augment_dataset_v3.py"
if v3_path.exists():
    spec = importlib.util.spec_from_file_location("augment_v3", str(v3_path))
    v3 = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(v3)
    except Exception:
        v3 = None

def post_transform(audio, sr, rng, meta, audio_parameters):
    # perform normalization similar to v3
    max_amplitude = audio_parameters.get('max_amplitude', 1.0)
    if audio_parameters.get('enable_normalization', False):
        peak = float(np.max(np.abs(audio)))
        if peak > max_amplitude and peak > 0:
            gain_factor = (max_amplitude / peak)
            try:
                gain_db = 20.0 * np.log10(gain_factor) if gain_factor > 0 else 0.0
            except Exception:
                gain_db = 0.0
            audio = audio * gain_factor
            return audio, {'normalization': {'peak_before': peak, 'gain_factor': gain_factor, 'gain_db': gain_db}}
    return audio, {}
