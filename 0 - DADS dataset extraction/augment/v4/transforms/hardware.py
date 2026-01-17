from pathlib import Path
import importlib.util

# load v3 mems sim if available
v3 = None
v3_path = Path(__file__).resolve().parents[3] / "0 - DADS dataset extraction" / "augment_dataset_v3.py"
if v3_path.exists():
    spec = importlib.util.spec_from_file_location("augment_v3", str(v3_path))
    v3 = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(v3)
    except Exception:
        v3 = None

def hardware_transform(audio, sr, rng, meta, mems_config):
    # apply mems simulation if enabled, else pass-through
    if mems_config.get('enabled', False):
        # sample deterministic hardware params from ranges if present
        try:
            import config as project_config
        except Exception:
            project_config = None

        chosen = dict(mems_config or {})
        # helper to pick a numeric from either a range key or list key
        def pick(key_base, rng, default=None):
            # e.g., highpass_cutoff_hz_range or highpass_cutoff_hz
            range_key = f"{key_base}_range"
            val = None
            if range_key in chosen and rng is not None:
                lo, hi = chosen[range_key]
                val = float(rng.uniform(float(lo), float(hi)))
            elif key_base in chosen and isinstance(chosen[key_base], (list, tuple)) and rng is not None:
                lo, hi = chosen[key_base][0], chosen[key_base][1]
                val = float(rng.uniform(float(lo), float(hi)))
            elif project_config and hasattr(project_config, 'HW_DOMAIN_DEFAULTS') and key_base in project_config.HW_DOMAIN_DEFAULTS:
                cfg = project_config.HW_DOMAIN_DEFAULTS.get(key_base if key_base.endswith('_range') else f"{key_base.replace('hpf','hpf')}_range", None)
                # fallback: try mapped names
                cfg = cfg or project_config.HW_DOMAIN_DEFAULTS.get(key_base)
                if isinstance(cfg, (list, tuple)) and rng is not None:
                    val = float(rng.uniform(float(cfg[0]), float(cfg[1])))
            return val if val is not None else default

        # pick highpass cutoff
        hp = pick('highpass_cutoff_hz', rng, default=chosen.get('highpass_cutoff_hz'))
        if hp is not None:
            chosen['highpass_cutoff_hz'] = float(hp)

        nf = pick('noise_floor_db', rng, default=chosen.get('noise_floor_db'))
        if nf is not None:
            chosen['noise_floor_db'] = float(nf)

        fr_tilt = pick('fr_tilt_db_per_oct', rng, default=chosen.get('fr_tilt_db_per_oct'))
        if fr_tilt is not None:
            chosen['fr_tilt_db_per_oct'] = float(fr_tilt)

        # notches
        # choose integer notch_count from range if provided
        if 'notch_count_range' in chosen and rng is not None:
            lo, hi = int(chosen['notch_count_range'][0]), int(chosen['notch_count_range'][1])
            chosen['notch_count'] = int(rng.integers(lo, hi + 1))
        elif 'notch_count' in chosen:
            chosen['notch_count'] = int(chosen['notch_count'])

        if 'notch_depth_db_range' in chosen and rng is not None:
            lo, hi = float(chosen['notch_depth_db_range'][0]), float(chosen['notch_depth_db_range'][1])
            chosen['notch_depth_db'] = float(rng.uniform(lo, hi))

        # adc bits
        if 'adc_bits' in chosen and isinstance(chosen['adc_bits'], (list, tuple)) and rng is not None:
            chosen['adc_bits'] = int(rng.choice(chosen['adc_bits']))

        # call v3 mems simulation with the concrete chosen config if available
        if v3 is not None and hasattr(v3, 'apply_mems_simulation'):
            audio2, mems_meta = v3.apply_mems_simulation(audio, sr, chosen, rng=rng)
            # ensure chosen params are recorded in returned metadata for audit
            for k in ('highpass_cutoff_hz', 'noise_floor_db', 'fr_tilt_db_per_oct', 'notch_count', 'notch_depth_db', 'adc_bits'):
                if k in chosen and k not in mems_meta:
                    mems_meta[k] = chosen[k]
            return audio2, mems_meta
        # fallback: no-op if v3 mems simulation is not present
        mems_meta = {k: chosen.get(k) for k in ('highpass_cutoff_hz', 'noise_floor_db', 'fr_tilt_db_per_oct', 'notch_count', 'notch_depth_db', 'adc_bits') if k in chosen}
        return audio, mems_meta
    return audio, {}
