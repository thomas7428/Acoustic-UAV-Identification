#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
# ensure repo root is on sys.path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location('p', str(Path(__file__).resolve().parents[1] / 'tests' / 'test_v4_propagation.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if __name__ == '__main__':
    mod.test_hf_rolloff_monotonic()
    mod.test_distance_attenuation_gain()
    print('RUN_V4_PROPAGATION_TEST: OK')
