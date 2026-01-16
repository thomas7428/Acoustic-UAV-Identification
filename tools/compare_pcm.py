#!/usr/bin/env python3
import json, hashlib
from pathlib import Path
import numpy as np
import soundfile as sf

run1 = Path("0 - DADS dataset extraction/dataset_smoke_run1")
run2 = Path("0 - DADS dataset extraction/dataset_smoke_run2")
meta = "augmentation_samples.jsonl"

meta1 = run1 / meta
lines = meta1.read_text(encoding='utf-8').splitlines()
rels = [json.loads(l).get('relpath') for l in lines]

header_only = 0
pcm_diff = 0

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

for rel in rels:
    p1 = run1 / rel
    p2 = run2 / rel
    b1 = p1.read_bytes()
    b2 = p2.read_bytes()
    if sha256_bytes(b1) == sha256_bytes(b2):
        continue

    x1, sr1 = sf.read(p1, dtype='float32', always_2d=False)
    x2, sr2 = sf.read(p2, dtype='float32', always_2d=False)

    if sr1 != sr2 or np.shape(x1) != np.shape(x2):
        print('SHAPE/SR MISMATCH:', rel, 'sr', sr1, sr2, 'shape', np.shape(x1), np.shape(x2))
        pcm_diff += 1
        continue

    h1 = hashlib.sha256(np.asarray(x1, dtype=np.float32).tobytes()).hexdigest()
    h2 = hashlib.sha256(np.asarray(x2, dtype=np.float32).tobytes()).hexdigest()

    if h1 == h2:
        header_only += 1
    else:
        d = float(np.max(np.abs(x1 - x2)))
        print('PCM DIFF:', rel, 'maxabs', d)
        pcm_diff += 1

print('=== SUMMARY ===')
print('total relpaths:', len(rels))
print('header_only_diffs:', header_only)
print('pcm_diffs:', pcm_diff)
