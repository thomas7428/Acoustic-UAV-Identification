"""
Enhanced Dataset Augmentation Script (v3.0 - PARALLEL)
Generates augmented audio samples for BOTH drone and no-drone classes:
- Drones: Mixed with background noise at various SNR levels
- No-drones: Complex background combinations with audio effects
- PARALLEL PROCESSING: Uses multiprocessing.Pool for x4-x8 speedup

Usage:
    python augment_dataset_v3.py --config augment_config_v4.json [--dry-run]
    AUGMENTATION_WORKERS=8 python augment_dataset_v3.py --config augment_config_v4.json
    
Features:
- Balanced augmentation for both classes
- Pitch shifting and time stretching for no-drones
- Multiple noise source mixing
- Amplitude variations and normalization
- Reproducible with random seed
- PARALLEL PROCESSING with configurable workers
"""

import os
import json
import argparse
# Suppress cryptography deprecation warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='paramiko')
warnings.filterwarnings('ignore', message='.*TripleDES.*')
warnings.filterwarnings('ignore', message='.*Blowfish.*')

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from multiprocessing import Pool
import random
from datetime import datetime
import sys

# Project config (centralized parameters and paths)
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as project_config

# Use shared audio distortion utilities
try:
    from tools.audio_distortions import apply_distortion_chain
except Exception:
    def apply_distortion_chain(audio, sr, conf):
        return audio

# Shared audio utilities (looping/crossfade/ensure_duration)
try:
    from tools.audio_utils import ensure_duration, loop_audio, crossfade, load_audio_file
except Exception:
    def ensure_duration(signal, sr, target_duration, crossfade_duration=0.1):
        target_samples = int(target_duration * sr)
        if len(signal) == target_samples:
            return signal
        elif len(signal) > target_samples:
            return signal[:target_samples]
        else:
            return np.pad(signal, (0, target_samples - len(signal)), mode='constant')
    
    def load_audio_file(file_path, sr, duration=None):
        try:
            signal, _ = librosa.load(file_path, sr=sr, duration=duration, mono=True)
            return signal
        except Exception:
            return None


def load_config(config_path):
    """Load augmentation configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return config_data


def calculate_snr(signal, noise):
    """Calculate actual SNR between signal and noise in dB."""
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    
    if noise_power == 0:
        return float('inf')
    
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else float('-inf')
    
    return snr_db


def apply_doppler_shift(signal, sr, shift_range=0.5):
    """Apply Doppler shift to simulate drone movement."""
    n_steps = np.random.uniform(-shift_range, shift_range)
    
    if abs(n_steps) > 0.05:
        shifted = librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)
        return shifted, float(n_steps)
    
    return signal, 0.0


def apply_intensity_modulation(signal, sr, mod_freq_range=(0.5, 2.0), depth=0.3):
    """Apply intensity modulation to simulate motor speed variations."""
    mod_freq = np.random.uniform(mod_freq_range[0], mod_freq_range[1])
    duration = len(signal) / sr
    t = np.linspace(0, duration, len(signal))
    envelope = 1.0 - depth + depth * np.sin(2 * np.pi * mod_freq * t)
    modulated = signal * envelope
    
    return modulated, float(mod_freq)


def apply_reverberation(signal, sr, delay_range=(50, 200), decay_range=(0.2, 0.4)):
    """Apply simple reverberation to simulate environment reflections."""
    delay_ms = np.random.randint(delay_range[0], delay_range[1])
    decay = np.random.uniform(decay_range[0], decay_range[1])
    delay_samples = int(delay_ms * sr / 1000)
    
    echo = np.zeros_like(signal)
    if delay_samples < len(signal):
        echo[delay_samples:] = signal[:-delay_samples] * decay
    
    reverb = signal + echo
    peak = np.max(np.abs(reverb))
    if peak > 1.0:
        reverb = reverb / peak
    
    return reverb, int(delay_ms), float(decay)


def apply_time_stretch_variation(signal, sr, rate_range=(0.95, 1.05)):
    """Apply time stretching to simulate rotor speed variations."""
    rate = np.random.uniform(rate_range[0], rate_range[1])
    
    if abs(rate - 1.0) > 0.01:
        stretched = librosa.effects.time_stretch(signal, rate=rate)
        return stretched, float(rate)
    
    return signal, 1.0


def apply_audio_effects(signal, sr, category_config):
    """Apply audio effects (pitch shift, time stretch) to signal."""
    metadata = {}
    processed = signal.copy()
    
    if category_config['enable_pitch_shift']:
        pitch_range = category_config['pitch_shift_range']
        n_steps = random.uniform(pitch_range[0], pitch_range[1])
        if abs(n_steps) > 0.1:
            processed = librosa.effects.pitch_shift(processed, sr=sr, n_steps=n_steps)
            metadata['pitch_shift_steps'] = float(n_steps)
    
    if category_config['enable_time_stretch']:
        stretch_range = category_config['time_stretch_range']
        rate = random.uniform(stretch_range[0], stretch_range[1])
        if abs(rate - 1.0) > 0.01:
            processed = librosa.effects.time_stretch(processed, rate=rate)
            metadata['time_stretch_rate'] = float(rate)
    
    return processed, metadata


def apply_mems_simulation(signal, sr, config):
    """
    Simulate MEMS microphone characteristics:
    - High-pass filter (typical MEMS roll-off)
    - Thermal noise floor
    - Soft clipping for high SPL
    
    Args:
        signal: Audio signal
        sr: Sample rate
        config: MEMS simulation config dict
    
    Returns:
        tuple: (processed_signal, metadata)
    """
    from scipy import signal as scipy_signal
    
    processed = signal.copy()
    metadata = {}
    
    # 1. High-pass filter (Butterworth 4th order)
    cutoff = config.get('highpass_cutoff_hz', 100)
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    
    if 0 < normalized_cutoff < 1:
        sos = scipy_signal.butter(4, normalized_cutoff, btype='high', output='sos')
        processed = scipy_signal.sosfilt(sos, processed)
        metadata['highpass_cutoff_hz'] = cutoff
    
    # 2. Add thermal noise floor
    noise_floor_db = config.get('noise_floor_db', -60)
    signal_power = np.mean(processed**2)
    if signal_power > 0:
        noise_power = signal_power / (10**(abs(noise_floor_db) / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(processed))
        processed = processed + noise
        metadata['noise_floor_db'] = noise_floor_db
    
    # 3. Soft clipping (tanh) to simulate SPL limit
    max_spl_db = config.get('max_spl_db', 94)
    # Convert dB SPL to normalized amplitude (simplified model)
    # 94 dB SPL ≈ 1.0 normalized amplitude
    clip_threshold = 10**((max_spl_db - 94) / 20)
    if clip_threshold > 0:
        processed = np.tanh(processed / clip_threshold) * clip_threshold
        metadata['max_spl_db'] = max_spl_db
    
    metadata['mems_applied'] = True
    return processed, metadata


def mix_drone_with_noise(drone_signal, noise_signals, target_snr_db, config, sr=None):
    """Mix drone audio with background noises at specified SNR with advanced augmentations."""
    metadata = {}
    augmented_drone = drone_signal.copy()

    if sr is None:
        sr = project_config.SAMPLE_RATE
    
    advanced_config = config['advanced_augmentations']

    if advanced_config['enabled']:
        if advanced_config['doppler_shift']['enabled'] and np.random.rand() < 0.5:
            shift_range = advanced_config['doppler_shift']['range']
            max_shift = max(abs(shift_range[0]), abs(shift_range[1]))
            augmented_drone, doppler = apply_doppler_shift(augmented_drone, sr, shift_range=max_shift)
            metadata['doppler_shift_semitones'] = doppler

        if advanced_config['intensity_modulation']['enabled'] and np.random.rand() < 0.4:
            freq_range = advanced_config['intensity_modulation']['freq_range']
            depth = advanced_config['intensity_modulation']['depth']
            augmented_drone, mod_freq = apply_intensity_modulation(augmented_drone, sr, 
                                                                    mod_freq_range=tuple(freq_range), 
                                                                    depth=depth)
            metadata['intensity_modulation_hz'] = mod_freq

        if advanced_config['reverberation']['enabled'] and np.random.rand() < 0.3:
            delay_range = advanced_config['reverberation']['delay_range']
            decay_range = advanced_config['reverberation']['decay_range']
            augmented_drone, delay, decay = apply_reverberation(augmented_drone, sr, 
                                                                 delay_range=tuple(delay_range), 
                                                                 decay_range=tuple(decay_range))
            metadata['reverb_delay_ms'] = delay
            metadata['reverb_decay'] = decay

        if advanced_config['time_stretch']['enabled'] and np.random.rand() < 0.2:
            rate_range = advanced_config['time_stretch']['rate_range']
            augmented_drone, stretch_rate = apply_time_stretch_variation(augmented_drone, sr, 
                                                                          rate_range=tuple(rate_range))
            metadata['time_stretch_rate'] = stretch_rate

            if len(augmented_drone) != len(drone_signal):
                if len(augmented_drone) > len(drone_signal):
                    augmented_drone = augmented_drone[:len(drone_signal)]
                else:
                    augmented_drone = np.pad(augmented_drone, (0, len(drone_signal) - len(augmented_drone)))
    
    combined_noise = np.zeros_like(augmented_drone)
    for noise in noise_signals:
        if len(noise) != len(augmented_drone):
            if len(noise) > len(augmented_drone):
                noise = noise[:len(augmented_drone)]
            else:
                noise = np.pad(noise, (0, len(augmented_drone) - len(noise)))
        combined_noise += noise
    combined_noise /= len(noise_signals)
    
    signal_power = np.mean(augmented_drone**2)
    noise_power = np.mean(combined_noise**2)
    
    if signal_power == 0 or noise_power == 0:
        return augmented_drone + combined_noise, 0, {"warning": "Zero power detected"}
    
    target_noise_power = signal_power / (10**(target_snr_db / 10))
    noise_scale_factor = np.sqrt(target_noise_power / noise_power)
    scaled_noise = combined_noise * noise_scale_factor
    
    if config['advanced']['enable_amplitude_variation']:
        variation_db = config['advanced']['amplitude_variation_db']
        variation_linear = np.random.uniform(-variation_db, variation_db)
        variation_factor = 10**(variation_linear / 20)
        scaled_noise *= variation_factor
    
    mixed = augmented_drone + scaled_noise
    
    # Apply MEMS simulation if enabled
    mems_config = config.get('mems_simulation', {})
    if mems_config.get('enabled', False):
        apply_prob = mems_config.get('apply_probability', 0.5)
        if np.random.rand() < apply_prob:
            mixed, mems_metadata = apply_mems_simulation(mixed, sr, mems_config)
            metadata.update(mems_metadata)
    
    max_amplitude = config['audio_parameters']['max_amplitude']
    if config['audio_parameters']['enable_normalization']:
        peak = np.max(np.abs(mixed))
        if peak > max_amplitude:
            mixed = mixed * (max_amplitude / peak)
    
    actual_snr = calculate_snr(augmented_drone, scaled_noise)
    
    metadata.update({
        "target_snr_db": target_snr_db,
        "actual_snr_db": float(actual_snr),
        "num_noise_sources": len(noise_signals),
        "normalized": config['audio_parameters']['enable_normalization'],
        "peak_amplitude": float(np.max(np.abs(mixed)))
    })
    
    return mixed, actual_snr, metadata


def mix_background_noises(noise_signals, amplitude_range, config, category_config, sr):
    """Mix multiple background noise signals with random amplitudes and effects."""
    if not noise_signals:
        return np.zeros(1), {"error": "No noise signals provided"}
    
    mixed = np.zeros_like(noise_signals[0])
    amplitudes = []
    
    for noise in noise_signals:
        amplitude = random.uniform(amplitude_range[0], amplitude_range[1])
        amplitudes.append(amplitude)
        mixed += noise * amplitude
    
    if len(noise_signals) > 1:
        mixed /= len(noise_signals)
    
    try:
        distorted = apply_distortion_chain(mixed, sr, category_config.get('augmentation_params', {}))
        mixed = distorted
        effects_metadata = {}
    except Exception:
        mixed, effects_metadata = apply_audio_effects(mixed, sr, category_config)
    
    # Apply MEMS simulation if enabled
    mems_config = config.get('mems_simulation', {})
    if mems_config.get('enabled', False):
        apply_prob = mems_config.get('apply_probability', 0.5)
        if np.random.rand() < apply_prob:
            mixed, mems_metadata = apply_mems_simulation(mixed, sr, mems_config)
            effects_metadata.update(mems_metadata)
    
    max_amplitude = config['audio_parameters']['max_amplitude']
    if config['audio_parameters']['enable_normalization']:
        peak = np.max(np.abs(mixed))
        if peak > max_amplitude:
            mixed = mixed * (max_amplitude / peak)
    
    metadata = {
        "num_sources": len(noise_signals),
        "amplitudes": amplitudes,
        "normalized": config['audio_parameters']['enable_normalization'],
        "peak_amplitude": float(np.max(np.abs(mixed))),
        **effects_metadata
    }
    
    return mixed, metadata


# ============================================================================
# PARALLEL WORKER FUNCTIONS
# ============================================================================

def _generate_one_drone_sample(args):
    """Worker function to generate one drone sample (for parallel processing)."""
    i, cat_name, category, drone_files, no_drone_files, output_dir, sr, duration, crossfade_duration, config, dry_run = args
    
    try:
        drone_file = random.choice(drone_files)
        drone_signal = load_audio_file(drone_file, sr)
        if drone_signal is None:
            return {'success': False}
        
        drone_signal = ensure_duration(drone_signal, sr, duration, crossfade_duration)
        
        num_noises = category['num_background_noises']
        noise_files = random.sample(no_drone_files, min(num_noises, len(no_drone_files)))
        
        noise_signals = []
        for noise_file in noise_files:
            noise_signal = load_audio_file(noise_file, sr)
            if noise_signal is not None:
                noise_signal = ensure_duration(noise_signal, sr, duration, crossfade_duration)
                noise_signals.append(noise_signal)
        
        if not noise_signals:
            return {'success': False}
        
        mixed_signal, actual_snr, mix_metadata = mix_drone_with_noise(
            drone_signal, noise_signals, category['snr_db'], config, sr=sr
        )
        
        if not dry_run:
            mixed_signal = ensure_duration(mixed_signal, sr, duration, crossfade_duration)
            mixed_signal = np.asarray(mixed_signal, dtype='float32')
            output_filename = f"aug_{cat_name}_{i:05d}.wav"
            output_path = output_dir / '1' / output_filename
            sf.write(str(output_path), mixed_signal, sr, subtype=project_config.AUDIO_WAV_SUBTYPE)
            
            return {
                'success': True, 'snr': actual_snr,
                'metadata': {
                    'filename': output_filename, 'category': cat_name, 'class': 'drone',
                    'label': category['label'], 'drone_source': drone_file.name,
                    'noise_sources': [nf.name for nf in noise_files], **mix_metadata
                }
            }
        return {'success': True, 'snr': actual_snr}
    except:
        return {'success': False}


def _generate_one_no_drone_sample(args):
    """Worker function to generate one no-drone sample (for parallel processing)."""
    i, cat_name, category, no_drone_files, output_dir, sr, duration, crossfade_duration, config, dry_run = args
    
    try:
        num_sources = category['num_noise_sources']
        noise_files = random.sample(no_drone_files, min(num_sources, len(no_drone_files)))
        
        noise_signals = []
        for noise_file in noise_files:
            noise_signal = load_audio_file(noise_file, sr)
            if noise_signal is not None:
                noise_signal = ensure_duration(noise_signal, sr, duration, crossfade_duration)
                noise_signals.append(noise_signal)
        
        if not noise_signals:
            return {'success': False}
        
        mixed_signal, mix_metadata = mix_background_noises(
            noise_signals, category['amplitude_range'], config, category, sr
        )
        
        if not dry_run:
            mixed_signal = ensure_duration(mixed_signal, sr, duration, crossfade_duration)
            mixed_signal = np.asarray(mixed_signal, dtype='float32')
            output_filename = f"aug_{cat_name}_{i:05d}.wav"
            output_path = output_dir / '0' / output_filename
            sf.write(str(output_path), mixed_signal, sr, subtype=project_config.AUDIO_WAV_SUBTYPE)
            
            return {
                'success': True,
                'metadata': {
                    'filename': output_filename, 'category': cat_name, 'class': 'no_drone',
                    'label': category['label'], 'noise_sources': [nf.name for nf in noise_files],
                    **mix_metadata
                }
            }
        return {'success': True}
    except:
        return {'success': False}


# ============================================================================
# GENERATION FUNCTIONS WITH PARALLEL PROCESSING
# ============================================================================

def generate_drone_augmented_samples(config, drone_files, no_drone_files, output_dir, sr, duration, crossfade_duration, dry_run=False):
    """Generate augmented drone samples (class 1) with PARALLEL PROCESSING."""
    
    if not config['drone_augmentation']['enabled']:
        print("\n[SKIP] Drone augmentation disabled in config")
        return {}
    
    print(f"\n{'='*80}")
    print("DRONE AUGMENTATION (Class 1) - PARALLEL MODE")
    print(f"{'='*80}")
    print(f"Workers: {project_config.AUGMENTATION_MAX_WORKERS}")
    
    stats = {
        'total_generated': 0,
        'categories': {},
        'errors': 0,
        'metadata': []
    }
    
    if not dry_run:
        (output_dir / '1').mkdir(parents=True, exist_ok=True)
    
    for category in config['drone_augmentation']['categories']:
        cat_name = category['name']
        samples_count = int(config['output']['samples_per_category_drone'] * category['proportion'])
        
        print(f"\n{'─'*80}")
        print(f"Category: {cat_name}")
        print(f"  SNR: {category['snr_db']} dB")
        print(f"  Background noises: {category['num_background_noises']}")
        print(f"  Samples to generate: {samples_count}")
        
        stats['categories'][cat_name] = {
            'target': samples_count,
            'generated': 0,
            'errors': 0,
            'avg_snr_achieved': []
        }
        
        # Generate samples in parallel
        progress_interval = max(1, samples_count // 5)
        task_args = [
            (i, cat_name, category, drone_files, no_drone_files, output_dir, sr, duration, crossfade_duration, config, dry_run)
            for i in range(samples_count)
        ]
        
        max_workers = min(project_config.AUGMENTATION_MAX_WORKERS, samples_count)
        with Pool(processes=max_workers) as pool:
            for idx, result in enumerate(pool.imap(_generate_one_drone_sample, task_args)):
                if idx % progress_interval == 0 or idx == samples_count - 1:
                    print(f"  {cat_name}: {idx+1}/{samples_count} ({(idx+1)/samples_count*100:.0f}%)", flush=True)
                
                if result['success']:
                    stats['categories'][cat_name]['generated'] += 1
                    if 'snr' in result:
                        stats['categories'][cat_name]['avg_snr_achieved'].append(result['snr'])
                    if 'metadata' in result:
                        stats['metadata'].append(result['metadata'])
                    stats['total_generated'] += 1
                else:
                    stats['categories'][cat_name]['errors'] += 1
                    stats['errors'] += 1
    
    # Calculate average SNR
    for cat_name, cat_stats in stats['categories'].items():
        if cat_stats['avg_snr_achieved']:
            avg_snr = np.mean(cat_stats['avg_snr_achieved'])
            cat_stats['avg_snr_db'] = float(avg_snr)
            del cat_stats['avg_snr_achieved']
    
    return stats


def generate_no_drone_augmented_samples(config, no_drone_files, output_dir, sr, duration, crossfade_duration, dry_run=False):
    """Generate augmented no-drone samples (class 0) with PARALLEL PROCESSING."""
    
    if not config['no_drone_augmentation']['enabled']:
        print("\n[SKIP] No-drone augmentation disabled in config")
        return {}
    
    print(f"\n{'='*80}")
    print("NO-DRONE AUGMENTATION (Class 0) - PARALLEL MODE")
    print(f"{'='*80}")
    print(f"Workers: {project_config.AUGMENTATION_MAX_WORKERS}")
    
    stats = {
        'total_generated': 0,
        'categories': {},
        'errors': 0,
        'metadata': []
    }
    
    if not dry_run:
        (output_dir / '0').mkdir(parents=True, exist_ok=True)
    
    for category in config['no_drone_augmentation']['categories']:
        cat_name = category['name']
        samples_count = int(config['output']['samples_per_category_no_drone'] * category['proportion'])
        
        print(f"\n{'─'*80}")
        print(f"Category: {cat_name}")
        print(f"  Noise sources: {category['num_noise_sources']}")
        print(f"  Amplitude range: {category['amplitude_range']}")
        print(f"  Pitch shift: {category['enable_pitch_shift']}")
        print(f"  Time stretch: {category['enable_time_stretch']}")
        print(f"  Samples to generate: {samples_count}")
        
        stats['categories'][cat_name] = {
            'target': samples_count,
            'generated': 0,
            'errors': 0
        }
        
        # Generate samples in parallel
        progress_interval = max(1, samples_count // 5)
        task_args = [
            (i, cat_name, category, no_drone_files, output_dir, sr, duration, crossfade_duration, config, dry_run)
            for i in range(samples_count)
        ]
        
        max_workers = min(project_config.AUGMENTATION_MAX_WORKERS, samples_count)
        with Pool(processes=max_workers) as pool:
            for idx, result in enumerate(pool.imap(_generate_one_no_drone_sample, task_args)):
                if idx % progress_interval == 0 or idx == samples_count - 1:
                    print(f"  {cat_name}: {idx+1}/{samples_count} ({(idx+1)/samples_count*100:.0f}%)", flush=True)
                
                if result['success']:
                    stats['categories'][cat_name]['generated'] += 1
                    if 'metadata' in result:
                        stats['metadata'].append(result['metadata'])
                    stats['total_generated'] += 1
                else:
                    stats['categories'][cat_name]['errors'] += 1
                    stats['errors'] += 1
    
    return stats


def generate_augmented_samples(aug_cfg, base_dir=None, dry_run=False):
    """
    Main entry point - Generate augmented audio samples with PARALLEL PROCESSING.
    
    Args:
        aug_cfg: Configuration dictionary
        base_dir: Base directory containing source datasets
        dry_run: If True, only print what would be done
    
    Returns:
        Dictionary with generation statistics
    """
    # Check if offline dataset exists and use it
    offline_dir = project_config.DATASET_DADS_OFFLINE_DIR
    use_offline = offline_dir.exists() and (offline_dir / "0").exists() and (offline_dir / "1").exists()
    
    if use_offline:
        print(f"\n✓ Found offline DADS dataset: {offline_dir}")
        print(f"  Using offline dataset as source for augmentation\n")
        drone_dir = offline_dir / '1'
        no_drone_dir = offline_dir / '0'
    else:
        # Use config-specified dirs or fallback to DATASET_DADS_DIR
        if aug_cfg.get('source_datasets') and aug_cfg['source_datasets'].get('drone_dir'):
            candidate = Path(aug_cfg['source_datasets']['drone_dir'])
            drone_dir = candidate if candidate.is_absolute() else Path(project_config.EXTRACTION_DIR) / candidate
        else:
            drone_dir = Path(project_config.DATASET_DADS_DIR) / '1'

        if aug_cfg.get('source_datasets') and aug_cfg['source_datasets'].get('no_drone_dir'):
            candidate = Path(aug_cfg['source_datasets']['no_drone_dir'])
            no_drone_dir = candidate if candidate.is_absolute() else Path(project_config.EXTRACTION_DIR) / candidate
        else:
            no_drone_dir = Path(project_config.DATASET_DADS_DIR) / '0'

    # Output directory
    if aug_cfg['output'].get('output_dir'):
        out_candidate = Path(aug_cfg['output']['output_dir'])
        output_dir = out_candidate if out_candidate.is_absolute() else Path(project_config.EXTRACTION_DIR) / out_candidate
    else:
        output_dir = Path(project_config.DATASET_AUGMENTED_DIR)

    # Audio params
    sr = project_config.SAMPLE_RATE
    duration = float(project_config.AUDIO_DURATION_S)
    crossfade_duration = aug_cfg['audio_parameters']['crossfade_duration_sec']

    # Random seed
    random.seed(aug_cfg['advanced']['random_seed'])
    np.random.seed(aug_cfg['advanced']['random_seed'])
    
    # Get source files
    drone_files = sorted(list(drone_dir.glob('*.wav'))) if drone_dir.exists() else []
    no_drone_files = sorted(list(no_drone_dir.glob('*.wav'))) if no_drone_dir.exists() else []
    
    print(f"\n{'='*80}")
    print("ENHANCED DATASET AUGMENTATION v3.0 (PARALLEL)")
    print(f"{'='*80}")
    print(f"Source drone samples (from {drone_dir}): {len(drone_files)}")
    print(f"Source no-drone samples (from {no_drone_dir}): {len(no_drone_files)}")
    print(f"Output directory: {output_dir}")
    print(f"Parallel workers: {project_config.AUGMENTATION_MAX_WORKERS}")
    
    if dry_run:
        print("\n[!] DRY RUN MODE - No files will be created\n")
    
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    drone_stats = generate_drone_augmented_samples(
        aug_cfg, drone_files, no_drone_files, output_dir, 
        sr, duration, crossfade_duration, dry_run
    )
    
    no_drone_stats = generate_no_drone_augmented_samples(
        aug_cfg, no_drone_files, output_dir,
        sr, duration, crossfade_duration, dry_run
    )
    
    # Combine statistics
    combined_stats = {
        'drone_augmentation': drone_stats,
        'no_drone_augmentation': no_drone_stats,
        'total_generated': drone_stats['total_generated'] + no_drone_stats['total_generated'],
        'total_errors': drone_stats['errors'] + no_drone_stats['errors']
    }
    
    # Save metadata
    if not dry_run and aug_cfg['output']['save_mixing_info']:
        all_metadata = drone_stats['metadata'] + no_drone_stats['metadata']
        raw_name = str(aug_cfg['output'].get('info_filename', 'augmentation_metadata.json')).strip()
        if not raw_name.lower().endswith('.json'):
            raw_name = raw_name + '.json'
        metadata_path = output_dir / raw_name

        tmp_path = metadata_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump({
                'generation_time': datetime.now().isoformat(),
                'version': '3.0-parallel',
                'workers': project_config.AUGMENTATION_MAX_WORKERS,
                'config': aug_cfg,
                'statistics': {
                    'drone_augmentation': {k: v for k, v in drone_stats.items() if k != 'metadata'},
                    'no_drone_augmentation': {k: v for k, v in no_drone_stats.items() if k != 'metadata'},
                    'total_generated': combined_stats['total_generated'],
                    'total_errors': combined_stats['total_errors']
                },
                'samples': all_metadata
            }, f, indent=2)
        
        try:
            tmp_path.replace(metadata_path)
        except Exception:
            import os
            os.replace(str(tmp_path), str(metadata_path))
        print(f"\n[OK] Metadata saved to: {metadata_path}")
    
    return combined_stats


def print_summary(stats):
    """Print generation summary."""
    print(f"\n{'='*80}")
    print("GENERATION SUMMARY")
    print(f"{'='*80}")
    
    if 'drone_augmentation' in stats and stats['drone_augmentation']:
        drone_stats = stats['drone_augmentation']
        print(f"\nDrone Augmentation (Class 1):")
        print(f"  Total generated: {drone_stats['total_generated']}")
        print(f"  Errors: {drone_stats['errors']}")
        
        if 'categories' in drone_stats:
            for cat_name, cat_stats in drone_stats['categories'].items():
                print(f"\n    {cat_name}:")
                print(f"      Generated: {cat_stats['generated']} / {cat_stats['target']}")
                if 'avg_snr_db' in cat_stats:
                    print(f"      Avg SNR achieved: {cat_stats['avg_snr_db']:.2f} dB")
    
    if 'no_drone_augmentation' in stats and stats['no_drone_augmentation']:
        no_drone_stats = stats['no_drone_augmentation']
        print(f"\nNo-Drone Augmentation (Class 0):")
        print(f"  Total generated: {no_drone_stats['total_generated']}")
        print(f"  Errors: {no_drone_stats['errors']}")
        
        if 'categories' in no_drone_stats:
            for cat_name, cat_stats in no_drone_stats['categories'].items():
                print(f"\n    {cat_name}:")
                print(f"      Generated: {cat_stats['generated']} / {cat_stats['target']}")
    
    print(f"\n{'─'*80}")
    print(f"TOTAL Generated: {stats['total_generated']}")
    print(f"TOTAL Errors: {stats['total_errors']}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced dataset augmentation v3.0 with PARALLEL PROCESSING",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with default workers (from config.py)
  python augment_dataset_v3.py --config augment_config_v4.json
  
  # Use 8 workers
  AUGMENTATION_WORKERS=8 python augment_dataset_v3.py --config augment_config_v4.json
  
  # Dry run
  python augment_dataset_v3.py --config augment_config_v4.json --dry-run
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to augmentation configuration JSON file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run without creating files')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return 1

    print(f"Loading configuration from: {config_path}")
    aug_cfg = load_config(config_path)

    stats = generate_augmented_samples(aug_cfg, dry_run=args.dry_run)
    print_summary(stats)
    
    if args.dry_run:
        print("[INFO] Dry run complete. Run without --dry-run to generate files.\n")
    else:
        print("[SUCCESS] Augmentation complete!\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
