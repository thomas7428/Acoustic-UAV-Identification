import json
from pathlib import Path
import soundfile as sf
import numpy as np

def write_wav(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure float32
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr)

def truncate_jsonl(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('', encoding='utf-8')

def append_jsonl_line(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def write_summary(path: Path, summary_obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(summary_obj, f, indent=2, ensure_ascii=False)
