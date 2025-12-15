"""
Deprecated wrapper for `build_mel_test_index.py`.

This file has been left in place as a small stub to avoid accidental
execution of the old logic. Use the canonical regenerating script instead:

  `1 - Preprocessing and Features Extraction/regenerate_mel_test_index_from_wavs.py`

Running this stub will print a deprecation message and exit with code 1.
"""

import sys


def main():
    print("[DEPRECATED] build_mel_test_index.py has been deprecated and moved to tools/deprecated/")
    print("Use: '1 - Preprocessing and Features Extraction/regenerate_mel_test_index_from_wavs.py' to create a canonical mel_test_index.json")
    return 1


if __name__ == '__main__':
    sys.exit(main())
