from pathlib import Path
import importlib.util
import numpy as np

# Dynamically load v3 helpers from their file path to avoid module name issues
v3_path = Path(__file__).resolve().parents[3] / "0 - DADS dataset extraction" / "augment_dataset_v3.py"
spec = importlib.util.spec_from_file_location("augment_v3", str(v3_path))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

def source_transform(sample_spec, sr, rng, meta):
    # sample_spec contains: type ('drone'|'no_drone'), category, index, file lists
    # For safety we expect sample_spec to include 'drone_file' or 'noise_files'
    if sample_spec.get('type') == 'drone':
        drone_file = sample_spec['drone_file']
        drone_signal = v3.load_audio_file(drone_file, sr)
        target_samples = int(sample_spec.get('duration_s', 1) * sr)
        drone_signal = v3.slice_random_segment(drone_signal, rng, target_samples)
        meta_delta = {'drone_source': getattr(drone_file, 'name', str(drone_file))}
        # load noise files if provided
        noise_bufs = []
        noise_files = sample_spec.get('noise_files', [])
        for nf in noise_files:
            s = v3.load_audio_file(nf, sr)
            if s is not None:
                noise_bufs.append(v3.slice_random_segment(s, rng, target_samples))
        if noise_bufs:
            meta_delta['noise_buffers'] = noise_bufs
            meta_delta['noise_sources'] = [getattr(nf, 'name', str(nf)) for nf in noise_files]
        return drone_signal, meta_delta
    else:
        # For no-drone return a mixed noise placeholder
        noises = []
        for nf in sample_spec.get('noise_files', []):
            s = v3.load_audio_file(nf, sr)
            if s is not None:
                noises.append(v3.slice_random_segment(s, rng, int(sample_spec.get('duration_s',1)*sr)))
        if not noises:
            return np.zeros(int(sample_spec.get('duration_s',1)*sr)), {'warning': 'no_noises'}
        # simple average
        mixed = np.zeros_like(noises[0])
        for n in noises:
            mixed += n
        mixed = mixed / len(noises)
        meta_delta = {'noise_sources': [getattr(nf,'name',str(nf)) for nf in sample_spec.get('noise_files',[])]}
        return mixed, meta_delta
