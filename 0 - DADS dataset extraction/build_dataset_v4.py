#!/usr/bin/env python3
"""Minimal wrapper for v4 dataset build (canonical entrypoint).

This wrapper treats `build_config.json` in the same directory as authoritative
and refuses CLI overrides. It delegates to `augment.generator.run_build`.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    base = Path(__file__).parent.resolve()
    config_path = base / "build_config.json"

    if not config_path.exists():
        print(f"ERROR: canonical config not found: {config_path}")
        return 2

    # Ensure local extraction-root augment package is importable
    sys.path.insert(0, str(base))

    try:
        from augment.generator import run_build
    except Exception as e:
        print("ERROR: failed to import augment.generator:", e)
        return 3

    # Always treat the local build_config.json as single truth; no overrides.
    return_code = run_build(str(config_path), out_dir=str(base), dry_run=True)
    return int(return_code)


if __name__ == "__main__":
    raise SystemExit(main())
