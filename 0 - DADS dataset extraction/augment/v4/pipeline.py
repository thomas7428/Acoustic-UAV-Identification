from typing import List, Callable, Tuple, Dict, Any

class Pipeline:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def run(self, audio, sr, rng, meta, ctx=None):
        # executes transforms sequentially, merging meta deltas
        meta_deltas = []
        cur = audio
        for t in self.transforms:
            cur, delta = t(cur, sr, rng, meta, **(ctx or {}))
            if delta:
                meta_deltas.append(delta)
        # merge
        merged = {}
        for d in meta_deltas:
            merged.update(d)
        return cur, merged
