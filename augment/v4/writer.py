from pathlib import Path
import json
import os
from typing import Optional

class SplitWriter:
    """Write audio files and an atomic append-only JSONL metadata stream per split.

    split_dir: path to <out_root>/<dataset_id>/<split>
    meta_name: filename for the jsonl stream inside the split dir
    """
    def __init__(self, split_dir: Path, meta_name: str = "augmentation_samples.jsonl"):
        self.split_dir = Path(split_dir)
        self.meta_name = meta_name
        self.split_dir.mkdir(parents=True, exist_ok=True)
        # ensure label subdirs exist on demand
        self._open_files = {}
        self._meta_path = self.split_dir / self.meta_name
        # create file if missing
        if not self._meta_path.exists():
            self._meta_path.write_text('', encoding='utf-8')

    def _ensure_label_dir(self, label: str):
        d = self.split_dir / str(label)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_wav(self, label: str, filename: str, audio_np, sr: int):
        """Write a wav file into <split_dir>/<label>/<filename>.

        `audio_np` should be a numpy array (float32 or similar).
        This method does not alter audio content.
        """
        import soundfile as sf
        d = self._ensure_label_dir(str(label))
        p = d / filename
        sf.write(str(p), audio_np, int(sr))
        return p

    def append_jsonl(self, train_meta: dict, debug_meta: Optional[dict] = None):
        """Atomically append one JSON object per line to the split's jsonl stream.

        Ensures data is flushed to disk.
        """
        line_obj = {'train_meta': train_meta}
        if debug_meta is not None:
            line_obj['debug_meta'] = debug_meta
        s = json.dumps(line_obj, ensure_ascii=False)
        # append atomically by opening, writing, flushing and fsync
        with open(self._meta_path, 'a', encoding='utf-8') as fh:
            fh.write(s + '\n')
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except Exception:
                pass
        return self._meta_path
