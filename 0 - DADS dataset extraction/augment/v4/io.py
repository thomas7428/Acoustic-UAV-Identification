import json
from pathlib import Path
import os
import soundfile as sf
import numpy as np


def write_wav_atomic(path: Path, audio: np.ndarray, sr: int, subtype: str = "PCM_16"):
    """Write `audio` to `path` atomically by writing to a same-dir tmp file then os.replace.

    tmp file uses the same suffix with an added .tmp to ensure same-filesystem replacement.
    Guarantees no leftover tmp files on success; on failure attempts best-effort cleanup.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        # ensure float32 for consistency; specify format so tmp name need not end with .wav
        sf.write(str(tmp), np.asarray(audio, dtype=np.float32), int(sr), format='WAV', subtype=subtype)
        os.replace(str(tmp), str(path))
    finally:
        # cleanup leftover tmp if anything failed before replace
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

def truncate_jsonl(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('', encoding='utf-8')

def append_jsonl_line(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    # append, flush and fsync to reduce risk of lost lines on crash
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

# backward compatible name
write_wav = write_wav_atomic

def write_summary(path: Path, summary_obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(summary_obj, f, indent=2, ensure_ascii=False)
