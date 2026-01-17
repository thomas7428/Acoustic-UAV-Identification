"""Modifier wrapper: RIR convolution modifier

Interface: apply_modifier(audio, sr, rng, meta, cfg) -> (audio_out, meta_delta)
"""
from augment.v4.transforms.rir import rir_convolution_transform


def apply_modifier(audio, sr, rng, meta, cfg):
    rir_cfg = cfg.get('rir', {})
    distance_m = cfg.get('distance_m')
    return rir_convolution_transform(audio, sr, rng, meta, rir_cfg, distance_m)
