#!/usr/bin/env python3
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('t', str(Path(__file__).resolve().parents[1] / 'tests' / 'test_v4_smoke.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if __name__ == '__main__':
    mod.test_v4_smoke_determinism(None)
    print('RUN_V4_SMOKE_TEST: OK')
