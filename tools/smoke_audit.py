import argparse, json, os, hashlib
from pathlib import Path
import numpy as np
import soundfile as sf

REQUIRED_KEYS = [
    "filename", "category", "class",
    "target_snr_db", "actual_snr_db_exported",
    "peak_dbfs", "rms_dbfs", "clip_count",
    "seed"
]


def load_meta(meta_path: Path):
    txt = meta_path.read_text(encoding="utf-8").strip()
    if not txt:
        raise RuntimeError(f"Empty metadata file: {meta_path}")

    # Try to parse as a full JSON document first; if that fails, fall back to JSONL line parsing
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and "samples" in obj:
            return obj["samples"]
        if isinstance(obj, list):
            return obj
        raise RuntimeError("Unknown JSON metadata schema")
    except Exception:
        samples = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
        return samples


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Augmentation output directory")
    ap.add_argument("--meta", default="augmentation_metadata.json", help="Metadata filename (json or jsonl)")
    ap.add_argument("--hash_wav", action="store_true", help="Compute sha256 for WAV files (slower)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    meta_path = out_dir / args.meta
    samples = load_meta(meta_path)

    n = len(samples)
    missing = {k: 0 for k in REQUIRED_KEYS}
    bad_files = 0

    by_cat = {}
    clip_total = 0
    gains = []
    deltas = []

    for s in samples:
        for k in REQUIRED_KEYS:
            if k == 'filename':
                # accept either 'filename' or 'relpath' produced by newer aug scripts
                if ('filename' not in s) and ('relpath' not in s):
                    missing[k] += 1
            else:
                if k not in s:
                    missing[k] += 1

        cat = s.get("category", "UNKNOWN")
        by_cat.setdefault(cat, {"n": 0, "deltas": [], "clips": 0})
        by_cat[cat]["n"] += 1

        t = s.get("target_snr_db", None)
        a = s.get("actual_snr_db_exported", None)
        if isinstance(t, (int, float)) and isinstance(a, (int, float)) and np.isfinite(t) and np.isfinite(a):
            d = float(a) - float(t)
            by_cat[cat]["deltas"].append(d)
            deltas.append(d)

        c = s.get("clip_count", 0) or 0
        clip_total += int(c)
        by_cat[cat]["clips"] += int(c)

        g = s.get("normalization_gain_db", None)
        if isinstance(g, (int, float)) and np.isfinite(g):
            gains.append(float(g))

        wav_key = s.get('filename') or s.get('relpath') or ''
        wav_path = out_dir / wav_key
        if wav_path.exists():
            try:
                x, sr = sf.read(wav_path, always_2d=False)
                if not np.all(np.isfinite(x)):
                    bad_files += 1
            except Exception:
                bad_files += 1
        else:
            bad_files += 1

    print(f"\n=== SMOKE AUDIT ===")
    print(f"Samples: {n}")
    print(f"Bad/missing WAV files or NaN audio: {bad_files}")
    print("\nMissing required keys:")
    for k, v in missing.items():
        if v:
            print(f"  {k}: {v}")

    if deltas:
        deltas_np = np.array(deltas)
        print("\nGlobal ΔSNR (actual-target) [dB]:")
        print(f"  mean={deltas_np.mean():.3f}  p50={np.percentile(deltas_np,50):.3f}  p95={np.percentile(deltas_np,95):.3f}  max|Δ|={np.max(np.abs(deltas_np)):.3f}")

    print(f"\nTotal clipped samples (sum clip_count): {clip_total}")

    if gains:
        gnp = np.array(gains)
        print("\nNormalization/anti-clip gain [dB] (should be ~0 most of the time):")
        print(f"  mean={gnp.mean():.3f}  p05={np.percentile(gnp,5):.3f}  p50={np.percentile(gnp,50):.3f}  p95={np.percentile(gnp,95):.3f}")

    print("\nPer-category ΔSNR summary:")
    for cat, d in sorted(by_cat.items(), key=lambda kv: kv[0]):
        dd = d["deltas"]
        if dd:
            dd = np.array(dd)
            print(f"  {cat:20s} n={d['n']:3d}  meanΔ={dd.mean():+.2f}  p95|Δ|={np.percentile(np.abs(dd),95):.2f}  clips={d['clips']}")
        else:
            print(f"  {cat:20s} n={d['n']:3d}  (no snr deltas)  clips={d['clips']}")

    if args.hash_wav:
        print("\nWAV hashes (sha256):")
        for s in samples[:20]:
            wav_key = s.get('filename') or s.get('relpath') or ''
            wav_path = out_dir / wav_key
            if wav_path.exists():
                print(f"  {wav_key}: {sha256_file(wav_path)[:16]}...")

if __name__ == "__main__":
    main()
