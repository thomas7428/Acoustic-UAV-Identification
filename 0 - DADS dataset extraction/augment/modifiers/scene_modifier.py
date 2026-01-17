"""Modifier wrapper: scene_mix modifier

Provides a small interface used by the augmentation engine:
  apply_modifier(audio, sr, rng, meta, cfg) -> (audio_out, meta_delta)

This wrapper calls the scene mix transform from the local `augment.v4.transforms`.
"""
from augment.v4.transforms.scene import scene_mix_transform
from pathlib import Path


def apply_modifier(audio, sr, rng, meta, cfg):
    # cfg expected to contain 'scene' sub-dict and 'noise_pool_dirs'
    scene_cfg = cfg.get('scene', {})
    noise_pool_dirs = cfg.get('scene', {}).get('noise_pool_dirs') or cfg.get('noise_pool_dirs')
    return scene_mix_transform(audio, sr, rng, meta, scene_cfg, noise_pool_dirs)
