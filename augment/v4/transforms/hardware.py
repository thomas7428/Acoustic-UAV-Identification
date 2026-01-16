from pathlib import Path
import importlib.util

# load v3 mems sim if available
v3_path = Path(__file__).resolve().parents[3] / "0 - DADS dataset extraction" / "augment_dataset_v3.py"
spec = importlib.util.spec_from_file_location("augment_v3", str(v3_path))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

def hardware_transform(audio, sr, rng, meta, mems_config):
    # apply mems simulation if enabled, else pass-through
    if mems_config.get('enabled', False):
        # v3.apply_mems_simulation returns (audio, metadata)
        audio2, mems_meta = v3.apply_mems_simulation(audio, sr, mems_config, rng=rng)
        return audio2, mems_meta
    return audio, {}
