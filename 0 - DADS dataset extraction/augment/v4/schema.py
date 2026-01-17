from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List

@dataclass
class SampleMeta:
    relpath: str
    seed_key: str
    seed: int
    label: int
    category: str
    target_snr_db: float = 0.0
    actual_snr_db_preexport: float = None
    peak_dbfs: float = None
    rms_dbfs: float = None
    clip_count: int = 0
    transforms: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # Keep this method but v4 runner uses explicit train/debug split
        d = asdict(self)
        d.update(self.extra)
        return d
