"""Lightweight mix transform used by the smoke runner.

Signature compatible with existing calls in `augment_dataset_v4.py`.
"""
import numpy as np


def mix_transform(audio, sr, rng, meta, category, cfg):
    # For smoke: no-op mixing beyond tagging metadata.
    meta_delta = {'mix_mode': 'noop', 'category': category.get('name') if isinstance(category, dict) else category}
    return audio, meta_delta
