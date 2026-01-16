import subprocess
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUN1 = ROOT / "0 - DADS dataset extraction" / "dataset_smoke_run1_v4"
RUN2 = ROOT / "0 - DADS dataset extraction" / "dataset_smoke_run2_v4"
CFG_BASE = ROOT / "0 - DADS dataset extraction" / "augment_config_smoke.json"

def make_cfg(out_dir: Path, seed=12345):
    cfg = json.loads(CFG_BASE.read_text(encoding='utf-8'))
    cfg.setdefault('advanced', {})
    cfg['advanced']['random_seed'] = seed
    cfg['advanced']['max_workers'] = 1
    cfg.setdefault('output', {})
    cfg['output']['output_dir'] = str(out_dir)
    cfg['output']['info_filename'] = 'augmentation_samples.jsonl'
    cfg['output']['summary_filename'] = 'augmentation_summary.json'
    return cfg

def run_once(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(out_dir)
    cfg_path = out_dir / 'cfg.json'
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    subprocess.check_call(["python3", "augment_dataset_v4.py", "--config", str(cfg_path), "--out_dir", str(out_dir)])

def test_v4_smoke_determinism(tmp_path):
    # Prepare
    if RUN1.exists():
        shutil.rmtree(RUN1)
    if RUN2.exists():
        shutil.rmtree(RUN2)
    RUN1.mkdir(parents=True)
    RUN2.mkdir(parents=True)

    run_once(RUN1)
    run_once(RUN2)

    # check counts
    wavs1 = list(RUN1.rglob('*.wav'))
    wavs2 = list(RUN2.rglob('*.wav'))
    assert len(wavs1) == len(wavs2)

    # validate jsonl lines
    meta1 = RUN1 / 'augmentation_samples.jsonl'
    meta2 = RUN2 / 'augmentation_samples.jsonl'
    lines1 = meta1.read_text(encoding='utf-8').splitlines()
    lines2 = meta2.read_text(encoding='utf-8').splitlines()
    assert len(lines1) == len(lines2) == len(wavs1)

    # use compare_runs (PCM-hash) to assert zero diffs
    subprocess.check_call(["python3", "-u", "tools/compare_runs.py", str(RUN1), str(RUN2), "--meta", "augmentation_samples.jsonl"]) 
