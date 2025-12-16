#!/usr/bin/env python3
"""Dataset validator
Scans dataset_train/dataset_val/dataset_test folders and extracted_features/*.json
Reports sample rate, channels, dtype, durations and mel frame stats.
Writes `tools/validate_report.json`.
"""
import os
import sys
import json
from pathlib import Path
from statistics import mean
import random

# Try import soundfile and librosa
try:
    import soundfile as sf
    import librosa
except Exception as e:
    print("Missing dependency: please install 'soundfile' and 'librosa' in the venv.")
    raise

# Add project root to path to import config
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

OUT_PATH = Path(__file__).parent / 'validate_report.json'

# Resolve dataset root (support DATASET_ROOT_OVERRIDE env used by master_setup)
DATASET_ROOT_OVERRIDE = os.environ.get('DATASET_ROOT_OVERRIDE')
if DATASET_ROOT_OVERRIDE:
    DATASET_ROOT = Path(DATASET_ROOT_OVERRIDE)
else:
    DATASET_ROOT = Path(getattr(config, 'DATASET_ROOT', Path(ROOT) / 'dataset'))

# Common dataset folders (master_setup uses dataset_train/... under project root)
CANDIDATES = [
    Path(config.PROJECT_ROOT) / '0 - DADS dataset extraction' / 'dataset_train',
    Path(config.PROJECT_ROOT) / '0 - DADS dataset extraction' / 'dataset_val',
    Path(config.PROJECT_ROOT) / '0 - DADS dataset extraction' / 'dataset_test',
    DATASET_ROOT / 'dataset_train',
    DATASET_ROOT / 'dataset_val',
    DATASET_ROOT / 'dataset_test',
]

# Unique candidates that exist
dataset_paths = [p for p in CANDIDATES if p.exists()]

report = {
    'datasets_scanned': [],
    'extracted_features': {},
    'summary': {}
}

# Helper to scan wav files in a folder (class subdirs '0' and '1')
def scan_dataset(path, sample_limit_per_class=300):
    stats = {
        'path': str(path),
        'classes': {},
        'total_files': 0
    }
    for class_label in ['0', '1']:
        class_path = Path(path) / class_label
        if not class_path.exists():
            continue
        wavs = sorted(list(class_path.glob('*.wav')))
        n = len(wavs)
        stats['classes'][class_label] = {
            'count': n,
            'sampled': 0,
            'sample_rates': [],
            'channels': [],
            'subtypes': [],
            'durations_s': []
        }
        # sample files to limit time
        sample_list = wavs if n <= sample_limit_per_class else random.sample(wavs, sample_limit_per_class)
        for wf in sample_list:
            try:
                info = sf.info(str(wf))
                sr = info.samplerate
                channels = info.channels
                subtype = info.subtype
                duration = info.frames / float(sr) if info.frames else None
                stats['classes'][class_label]['sampled'] += 1
                stats['classes'][class_label]['sample_rates'].append(sr)
                stats['classes'][class_label]['channels'].append(channels)
                stats['classes'][class_label]['subtypes'].append(subtype)
                stats['classes'][class_label]['durations_s'].append(duration)
            except Exception as e:
                # record error
                stats.setdefault('errors', []).append({'file': str(wf), 'error': str(e)})
        stats['total_files'] += n
    return stats

# Scan datasets
for p in dataset_paths:
    r = scan_dataset(p)
    report['datasets_scanned'].append(r)

# Scan extracted_features JSONs
EX_DIR = Path(config.EXTRACTED_FEATURES_DIR)
if EX_DIR.exists():
    for fname in EX_DIR.glob('*.json'):
        try:
            data = json.load(open(fname, 'r'))
        except Exception as e:
            report['extracted_features'][fname.name] = {'error': str(e)}
            continue
        summary = {}
        # mel entries
        if 'mel' in data:
            mels = data.get('mel', [])
            if isinstance(mels, list) and len(mels) > 0:
                # assume list of lists-of-lists
                frames = [len(m[0]) if (isinstance(m, list) and len(m)>0 and isinstance(m[0], list)) else None for m in mels]
                frames = [f for f in frames if f is not None]
                summary['mel_count'] = len(mels)
                summary['mel_time_frames_unique'] = sorted(list(set(frames))) if frames else []
                summary['mel_time_frames_sample'] = frames[:10]
            else:
                summary['mel_count'] = 0
        # mfcc entries
        if 'mfcc' in data:
            mfccs = data.get('mfcc', [])
            summary['mfcc_count'] = len(mfccs)
        report['extracted_features'][fname.name] = summary
else:
    report['extracted_features'] = {'error': f'extracted_features dir not found: {EX_DIR}'}

# Aggregate quick summary
summary = {
    'total_datasets_paths_found': len(dataset_paths),
    'extracted_features_found': len(report['extracted_features'])
}
report['summary'] = summary

# Save report
with open(OUT_PATH, 'w') as fp:
    json.dump(report, fp, indent=2)

# Print concise human summary
print("\nDataset validation report written to:", OUT_PATH)
print("Datasets scanned:")
for d in report['datasets_scanned']:
    print(f" - {d['path']}: total_files={d['total_files']}")
    for cl, cinfo in d['classes'].items():
        if cinfo['sampled'] == 0:
            print(f"    class {cl}: no files sampled")
            continue
        sr_set = sorted(set(cinfo['sample_rates']))
        channels_set = sorted(set(cinfo['channels']))
        subtype_set = sorted(set(cinfo['subtypes']))
        dur_list = [x for x in cinfo['durations_s'] if x is not None]
        print(f"    class {cl}: sampled={cinfo['sampled']}, sample_rates={sr_set}, channels={channels_set}, subtypes={subtype_set}")
        if dur_list:
            print(f"      durations s: min={min(dur_list):.3f}, max={max(dur_list):.3f}, mean={mean(dur_list):.3f}")

print("\nExtracted features summary:")
for k,v in report['extracted_features'].items():
    print(f" - {k}: {v}")

print('\nDone.')
