"""
Generate a small set of audio examples (WAV + waveform + spectrogram + HTML)

This script picks four representative audio files and copies them to
`6 - Visualization/outputs/audio_examples/`:
  - extreme ambient
  - normal ambient
  - extreme far drone (700m / 600m)
  - close drone (50m / 100m)

It creates waveform and spectrogram PNGs for each file and an `index.html`
with embedded `<audio>` players for quick inspection.

Usage:
  python audio_examples.py

"""
import os
from pathlib import Path
import shutil
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent
BASE_DATASET = PROJECT_ROOT / "0 - DADS dataset extraction"
SEARCH_DIRS = [
    BASE_DATASET / 'dataset_augmented',
    BASE_DATASET / 'dataset_combined',
    BASE_DATASET / 'dataset_test',
]

OUTPUT_DIR = Path(__file__).parent / 'outputs' / 'audio_examples'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_first_match(patterns):
    for d in SEARCH_DIRS:
        if not d.exists():
            continue
        for root, _, files in os.walk(d):
            for f in files:
                fname = f.lower()
                for pat in patterns:
                    if pat in fname:
                        return Path(root) / f
    return None


def pick_examples():
    picks = {}

    # extreme ambient
    picks['ambient_extreme'] = find_first_match(['ambient_extreme', 'ambient_extreme'])

    # normal ambient (fallback to moderate/complex)
    picks['ambient_normal'] = find_first_match(['ambient_moderate', 'ambient_complex', 'ambient_simple'])

    # extreme far drone (prefer 700m then 600m)
    picks['drone_far'] = find_first_match(['700m', '600m', 'very_far'])

    # close drone (prefer 50m then 100m)
    picks['drone_close'] = find_first_match(['50m', '100m', 'close'])

    return picks


def make_waveform_and_spectrogram(src_path, dst_dir, sr=22050):
    y, _ = librosa.load(src_path, sr=sr, mono=True)
    name = src_path.stem

    # Waveform
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f'Waveform: {name}')
    plt.tight_layout()
    wf_path = dst_dir / f'{name}_waveform.png'
    plt.savefig(wf_path, dpi=150)
    plt.close()

    # Spectrogram (Mel)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(6, 3))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel spectrogram: {name}')
    plt.tight_layout()
    spec_path = dst_dir / f'{name}_spectrogram.png'
    plt.savefig(spec_path, dpi=150)
    plt.close()

    return wf_path.name, spec_path.name


def build_html(entries, out_dir):
    html = ['<html><head><meta charset="utf-8"><title>Audio examples</title></head><body>']
    html.append('<h1>Audio Examples</h1>')
    for key, info in entries.items():
        html.append(f'<h2>{key}</h2>')
        if info is None:
            html.append('<p><em>Not found</em></p>')
            continue
        fname = info['file'].name
        html.append(f'<audio controls src="{fname}"></audio>')
        html.append('<br/>')
        html.append(f'<img src="{info["waveform"]}" style="max-width:800px;"><br/>')
        html.append(f'<img src="{info["spectrogram"]}" style="max-width:800px;"><br/>')
    html.append('</body></html>')

    with open(out_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))


def main():
    picks = pick_examples()
    entries = {}

    for key, path in picks.items():
        if path is None:
            entries[key] = None
            continue

        dst_wav = OUTPUT_DIR / path.name
        if not dst_wav.exists():
            shutil.copy2(path, dst_wav)

        wf_name, spec_name = make_waveform_and_spectrogram(path, OUTPUT_DIR)

        entries[key] = {
            'file': dst_wav,
            'waveform': wf_name,
            'spectrogram': spec_name
        }

    build_html(entries, OUTPUT_DIR)
    print(f"[OK] Audio examples created in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
