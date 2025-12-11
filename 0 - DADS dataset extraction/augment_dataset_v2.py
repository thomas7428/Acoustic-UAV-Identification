"""
Enhanced Dataset Augmentation Script (v2.0)
Generates augmented audio samples for BOTH drone and no-drone classes:
- Drones: Mixed with background noise at various SNR levels
- No-drones: Complex background combinations with audio effects

Usage:
    python augment_dataset_v2.py [--config augment_config_v2.json] [--dry-run]
    
Features:
- Balanced augmentation for both classes
- Pitch shifting and time stretching for no-drones
- Multiple noise source mixing
- Amplitude variations and normalization
- Reproducible with random seed
"""

import os
import json
import argparse
import warnings

# Suppress cryptography deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='paramiko')
warnings.filterwarnings('ignore', message='.*TripleDES.*')
warnings.filterwarnings('ignore', message='.*Blowfish.*')

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import random
from datetime import datetime


def load_config(config_path):
    """Load augmentation configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def crossfade(signal1, signal2, fade_duration_samples):
    """
    Apply crossfade between two audio signals.
    
    Args:
        signal1: First audio signal (numpy array)
        signal2: Second audio signal (numpy array)
        fade_duration_samples: Number of samples for crossfade
    
    Returns:
        Crossfaded signal
    """
    if fade_duration_samples == 0 or len(signal1) < fade_duration_samples:
        return np.concatenate([signal1, signal2])
    
    # Create fade curves
    fade_out = np.linspace(1, 0, fade_duration_samples)
    fade_in = np.linspace(0, 1, fade_duration_samples)
    
    # Apply crossfade to overlapping region
    end_part = signal1[-fade_duration_samples:] * fade_out
    start_part = signal2[:fade_duration_samples] * fade_in
    crossfaded_part = end_part + start_part
    
    # Combine signals
    result = np.concatenate([
        signal1[:-fade_duration_samples],
        crossfaded_part,
        signal2[fade_duration_samples:]
    ])
    
    return result


def loop_audio(signal, target_duration, sr, crossfade_duration=0.1):
    """
    Loop audio signal to reach target duration with smooth crossfade transitions.
    
    Args:
        signal: Audio signal (numpy array)
        target_duration: Target duration in seconds
        sr: Sample rate
        crossfade_duration: Duration of crossfade in seconds
    
    Returns:
        Looped audio signal
    """
    target_samples = int(target_duration * sr)
    signal_length = len(signal)
    
    # If signal is already long enough, just trim
    if signal_length >= target_samples:
        return signal[:target_samples]
    
    # Calculate crossfade samples
    crossfade_samples = int(crossfade_duration * sr)
    crossfade_samples = min(crossfade_samples, signal_length // 4)  # Max 25% of signal length
    
    # Loop the signal with crossfade
    result = signal.copy()
    
    while len(result) < target_samples:
        if len(result) + signal_length - crossfade_samples <= target_samples:
            # Add full signal with crossfade
            result = crossfade(result, signal, crossfade_samples)
        else:
            # Add partial signal to reach exact duration
            remaining = target_samples - len(result)
            if remaining > crossfade_samples:
                result = crossfade(result, signal[:remaining], crossfade_samples)
            else:
                # Not enough space for crossfade, simple concatenation
                result = np.concatenate([result, signal[:remaining]])
    
    return result[:target_samples]


def calculate_snr(signal, noise):
    """Calculate actual SNR between signal and noise in dB."""
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    
    if noise_power == 0:
        return float('inf')
    
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else float('-inf')
    
    return snr_db


def apply_audio_effects(signal, sr, category_config):
    """
    Apply audio effects (pitch shift, time stretch) to signal.
    
    Args:
        signal: Audio signal (numpy array)
        sr: Sample rate
        category_config: Category configuration dict with effect settings
    
    Returns:
        tuple: (processed_signal, effects_metadata)
    """
    metadata = {}
    processed = signal.copy()
    
    # Pitch shifting
    if category_config.get('enable_pitch_shift', False):
        pitch_range = category_config.get('pitch_shift_range', [-1, 1])
        n_steps = random.uniform(pitch_range[0], pitch_range[1])
        if abs(n_steps) > 0.1:  # Only apply if significant
            processed = librosa.effects.pitch_shift(processed, sr=sr, n_steps=n_steps)
            metadata['pitch_shift_steps'] = float(n_steps)
    
    # Time stretching
    if category_config.get('enable_time_stretch', False):
        stretch_range = category_config.get('time_stretch_range', [0.9, 1.1])
        rate = random.uniform(stretch_range[0], stretch_range[1])
        if abs(rate - 1.0) > 0.01:  # Only apply if significant
            processed = librosa.effects.time_stretch(processed, rate=rate)
            metadata['time_stretch_rate'] = float(rate)
    
    return processed, metadata


def mix_drone_with_noise(drone_signal, noise_signals, target_snr_db, config):
    """
    Mix drone audio with background noises at specified SNR.
    
    Args:
        drone_signal: Drone audio (numpy array)
        noise_signals: List of noise audio signals (list of numpy arrays)
        target_snr_db: Target Signal-to-Noise Ratio in dB
        config: Configuration dict with audio parameters
    
    Returns:
        tuple: (mixed_signal, actual_snr_db, metadata)
    """
    # Combine multiple noise signals (average)
    combined_noise = np.zeros_like(drone_signal)
    for noise in noise_signals:
        combined_noise += noise
    combined_noise /= len(noise_signals)
    
    # Calculate powers
    signal_power = np.mean(drone_signal**2)
    noise_power = np.mean(combined_noise**2)
    
    # Avoid division by zero
    if signal_power == 0 or noise_power == 0:
        return drone_signal + combined_noise, 0, {"warning": "Zero power detected"}
    
    # Calculate target noise power based on desired SNR
    # SNR (dB) = 10 * log10(P_signal / P_noise)
    # => P_noise = P_signal / 10^(SNR/10)
    target_noise_power = signal_power / (10**(target_snr_db / 10))
    
    # Scale noise to achieve target SNR
    noise_scale_factor = np.sqrt(target_noise_power / noise_power)
    scaled_noise = combined_noise * noise_scale_factor
    
    # Optional: Add slight amplitude variation
    if config['advanced']['enable_amplitude_variation']:
        variation_db = config['advanced']['amplitude_variation_db']
        variation_linear = np.random.uniform(-variation_db, variation_db)
        variation_factor = 10**(variation_linear / 20)
        scaled_noise *= variation_factor
    
    # Mix signals
    mixed = drone_signal + scaled_noise
    
    # Normalize to prevent clipping
    max_amplitude = config['audio_parameters']['max_amplitude']
    if config['audio_parameters']['enable_normalization']:
        peak = np.max(np.abs(mixed))
        if peak > max_amplitude:
            mixed = mixed * (max_amplitude / peak)
    
    # Calculate actual achieved SNR
    actual_snr = calculate_snr(drone_signal, scaled_noise)
    
    # Metadata
    metadata = {
        "target_snr_db": target_snr_db,
        "actual_snr_db": float(actual_snr),
        "num_noise_sources": len(noise_signals),
        "normalized": config['audio_parameters']['enable_normalization'],
        "peak_amplitude": float(np.max(np.abs(mixed)))
    }
    
    return mixed, actual_snr, metadata


def mix_background_noises(noise_signals, amplitude_range, config, category_config, sr):
    """
    Mix multiple background noise signals with random amplitudes and effects.
    For no-drone class augmentation.
    
    Args:
        noise_signals: List of noise audio signals
        amplitude_range: [min, max] amplitude scaling factors
        config: Global configuration dict
        category_config: Category-specific configuration
        sr: Sample rate
    
    Returns:
        tuple: (mixed_signal, metadata)
    """
    if not noise_signals:
        return np.zeros(1), {"error": "No noise signals provided"}
    
    # Mix noises with random amplitude weights
    mixed = np.zeros_like(noise_signals[0])
    amplitudes = []
    
    for noise in noise_signals:
        amplitude = random.uniform(amplitude_range[0], amplitude_range[1])
        amplitudes.append(amplitude)
        mixed += noise * amplitude
    
    # Normalize total amplitude
    if len(noise_signals) > 1:
        mixed /= len(noise_signals)
    
    # Apply audio effects (pitch shift, time stretch)
    mixed, effects_metadata = apply_audio_effects(mixed, sr, category_config)
    
    # Normalize to prevent clipping
    max_amplitude = config['audio_parameters']['max_amplitude']
    if config['audio_parameters']['enable_normalization']:
        peak = np.max(np.abs(mixed))
        if peak > max_amplitude:
            mixed = mixed * (max_amplitude / peak)
    
    # Metadata
    metadata = {
        "num_sources": len(noise_signals),
        "amplitudes": amplitudes,
        "normalized": config['audio_parameters']['enable_normalization'],
        "peak_amplitude": float(np.max(np.abs(mixed))),
        **effects_metadata
    }
    
    return mixed, metadata


def load_audio_file(file_path, sr, duration=None):
    """Load audio file with error handling."""
    try:
        signal, _ = librosa.load(file_path, sr=sr, duration=duration)
        return signal
    except Exception as e:
        # Silent failure for cleaner output
        return None


def ensure_duration(signal, sr, target_duration, crossfade_duration=0.1):
    """Ensure audio signal has exact target duration (loop or pad as needed)."""
    target_samples = int(target_duration * sr)
    
    if len(signal) == target_samples:
        return signal
    elif len(signal) > target_samples:
        return signal[:target_samples]
    else:
        # Loop with crossfade
        return loop_audio(signal, target_duration, sr, crossfade_duration)


def generate_drone_augmented_samples(config, drone_files, no_drone_files, output_dir, sr, duration, crossfade_duration, dry_run=False):
    """Generate augmented drone samples (class 1) mixed with background noise."""
    
    if not config['drone_augmentation']['enabled']:
        print("\n[SKIP] Drone augmentation disabled in config")
        return {}
    
    print(f"\n{'='*80}")
    print("DRONE AUGMENTATION (Class 1)")
    print(f"{'='*80}")
    
    stats = {
        'total_generated': 0,
        'categories': {},
        'errors': 0,
        'metadata': []
    }
    
    # Create output directory
    if not dry_run:
        (output_dir / '1').mkdir(parents=True, exist_ok=True)
    
    # Generate samples for each drone category
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
        
        # Generate samples
        for i in tqdm(range(samples_count), desc=f"Generating {cat_name}"):
            try:
                # Randomly select drone sample
                drone_file = random.choice(drone_files)
                drone_signal = load_audio_file(drone_file, sr)
                if drone_signal is None:
                    stats['categories'][cat_name]['errors'] += 1
                    continue
                
                # Ensure target duration
                drone_signal = ensure_duration(drone_signal, sr, duration, crossfade_duration)
                
                # Randomly select background noise samples
                num_noises = category['num_background_noises']
                noise_files = random.sample(no_drone_files, min(num_noises, len(no_drone_files)))
                
                noise_signals = []
                for noise_file in noise_files:
                    noise_signal = load_audio_file(noise_file, sr)
                    if noise_signal is not None:
                        noise_signal = ensure_duration(noise_signal, sr, duration, crossfade_duration)
                        noise_signals.append(noise_signal)
                
                if not noise_signals:
                    stats['categories'][cat_name]['errors'] += 1
                    continue
                
                # Mix drone with noise at target SNR
                mixed_signal, actual_snr, mix_metadata = mix_drone_with_noise(
                    drone_signal,
                    noise_signals,
                    category['snr_db'],
                    config
                )
                
                # Save mixed audio
                if not dry_run:
                    output_filename = f"aug_{cat_name}_{i:05d}.wav"
                    output_path = output_dir / '1' / output_filename
                    sf.write(output_path, mixed_signal, sr)
                    
                    # Save metadata
                    sample_metadata = {
                        'filename': output_filename,
                        'category': cat_name,
                        'class': 'drone',
                        'label': category['label'],
                        'drone_source': drone_file.name,
                        'noise_sources': [nf.name for nf in noise_files],
                        **mix_metadata
                    }
                    stats['metadata'].append(sample_metadata)
                
                stats['categories'][cat_name]['generated'] += 1
                stats['categories'][cat_name]['avg_snr_achieved'].append(actual_snr)
                stats['total_generated'] += 1
                
            except Exception as e:
                stats['categories'][cat_name]['errors'] += 1
                stats['errors'] += 1
    
    # Calculate average achieved SNR per category
    for cat_name, cat_stats in stats['categories'].items():
        if cat_stats['avg_snr_achieved']:
            avg_snr = np.mean(cat_stats['avg_snr_achieved'])
            cat_stats['avg_snr_db'] = float(avg_snr)
            del cat_stats['avg_snr_achieved']
    
    return stats


def generate_no_drone_augmented_samples(config, no_drone_files, output_dir, sr, duration, crossfade_duration, dry_run=False):
    """Generate augmented no-drone samples (class 0) with complex backgrounds."""
    
    if not config['no_drone_augmentation']['enabled']:
        print("\n[SKIP] No-drone augmentation disabled in config")
        return {}
    
    print(f"\n{'='*80}")
    print("NO-DRONE AUGMENTATION (Class 0)")
    print(f"{'='*80}")
    
    stats = {
        'total_generated': 0,
        'categories': {},
        'errors': 0,
        'metadata': []
    }
    
    # Create output directory
    if not dry_run:
        (output_dir / '0').mkdir(parents=True, exist_ok=True)
    
    # Generate samples for each no-drone category
    for category in config['no_drone_augmentation']['categories']:
        cat_name = category['name']
        samples_count = int(config['output']['samples_per_category_no_drone'] * category['proportion'])
        
        print(f"\n{'─'*80}")
        print(f"Category: {cat_name}")
        print(f"  Noise sources: {category['num_noise_sources']}")
        print(f"  Amplitude range: {category['amplitude_range']}")
        print(f"  Pitch shift: {category.get('enable_pitch_shift', False)}")
        print(f"  Time stretch: {category.get('enable_time_stretch', False)}")
        print(f"  Samples to generate: {samples_count}")
        
        stats['categories'][cat_name] = {
            'target': samples_count,
            'generated': 0,
            'errors': 0
        }
        
        # Generate samples
        for i in tqdm(range(samples_count), desc=f"Generating {cat_name}"):
            try:
                # Randomly select noise sources
                num_sources = category['num_noise_sources']
                noise_files = random.sample(no_drone_files, min(num_sources, len(no_drone_files)))
                
                noise_signals = []
                for noise_file in noise_files:
                    noise_signal = load_audio_file(noise_file, sr)
                    if noise_signal is not None:
                        noise_signal = ensure_duration(noise_signal, sr, duration, crossfade_duration)
                        noise_signals.append(noise_signal)
                
                if not noise_signals:
                    stats['categories'][cat_name]['errors'] += 1
                    continue
                
                # Mix background noises with effects
                mixed_signal, mix_metadata = mix_background_noises(
                    noise_signals,
                    category['amplitude_range'],
                    config,
                    category,
                    sr
                )
                
                # Save mixed audio
                if not dry_run:
                    output_filename = f"aug_{cat_name}_{i:05d}.wav"
                    output_path = output_dir / '0' / output_filename
                    sf.write(output_path, mixed_signal, sr)
                    
                    # Save metadata
                    sample_metadata = {
                        'filename': output_filename,
                        'category': cat_name,
                        'class': 'no_drone',
                        'label': category['label'],
                        'noise_sources': [nf.name for nf in noise_files],
                        **mix_metadata
                    }
                    stats['metadata'].append(sample_metadata)
                
                stats['categories'][cat_name]['generated'] += 1
                stats['total_generated'] += 1
                
            except Exception as e:
                stats['categories'][cat_name]['errors'] += 1
                stats['errors'] += 1
    
    return stats


def generate_augmented_samples(config, base_dir, dry_run=False):
    """
    Generate augmented audio samples for both drone and no-drone classes.
    
    Args:
        config: Configuration dictionary
        base_dir: Base directory containing source datasets
        dry_run: If True, only print what would be done without creating files
    
    Returns:
        Dictionary with generation statistics
    """
    # Setup paths
    base_path = Path(base_dir)
    drone_dir = base_path / config['source_datasets']['drone_dir']
    no_drone_dir = base_path / config['source_datasets']['no_drone_dir']
    output_dir = base_path / config['output']['output_dir']
    
    # Get audio parameters
    sr = config['audio_parameters']['sample_rate']
    duration = config['audio_parameters']['target_duration_sec']
    crossfade_duration = config['audio_parameters']['crossfade_duration_sec']
    
    # Set random seed for reproducibility
    random.seed(config['advanced']['random_seed'])
    np.random.seed(config['advanced']['random_seed'])
    
    # Get list of source files
    drone_files = sorted(list(drone_dir.glob('*.wav')))
    no_drone_files = sorted(list(no_drone_dir.glob('*.wav')))
    
    print(f"\n{'='*80}")
    print("ENHANCED DATASET AUGMENTATION v2.0")
    print(f"{'='*80}")
    print(f"Source drone samples: {len(drone_files)}")
    print(f"Source no-drone samples: {len(no_drone_files)}")
    print(f"Output directory: {output_dir}")
    
    if dry_run:
        print("\n[!] DRY RUN MODE - No files will be created\n")
    
    # Create output directory structure
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate drone augmented samples (class 1)
    drone_stats = generate_drone_augmented_samples(
        config, drone_files, no_drone_files, output_dir, 
        sr, duration, crossfade_duration, dry_run
    )
    
    # Generate no-drone augmented samples (class 0)
    no_drone_stats = generate_no_drone_augmented_samples(
        config, no_drone_files, output_dir,
        sr, duration, crossfade_duration, dry_run
    )
    
    # Combine statistics
    combined_stats = {
        'drone_augmentation': drone_stats,
        'no_drone_augmentation': no_drone_stats,
        'total_generated': drone_stats.get('total_generated', 0) + no_drone_stats.get('total_generated', 0),
        'total_errors': drone_stats.get('errors', 0) + no_drone_stats.get('errors', 0)
    }
    
    # Save combined metadata to JSON
    if not dry_run and config['output']['save_mixing_info']:
        all_metadata = drone_stats.get('metadata', []) + no_drone_stats.get('metadata', [])
        metadata_path = output_dir / config['output']['info_filename']
        
        with open(metadata_path, 'w') as f:
            json.dump({
                'generation_time': datetime.now().isoformat(),
                'version': '2.0',
                'config': config,
                'statistics': {
                    'drone_augmentation': {k: v for k, v in drone_stats.items() if k != 'metadata'},
                    'no_drone_augmentation': {k: v for k, v in no_drone_stats.items() if k != 'metadata'},
                    'total_generated': combined_stats['total_generated'],
                    'total_errors': combined_stats['total_errors']
                },
                'samples': all_metadata
            }, f, indent=2)
        print(f"\n[OK] Metadata saved to: {metadata_path}")
    
    return combined_stats


def print_summary(stats):
    """Print generation summary."""
    print(f"\n{'='*80}")
    print("GENERATION SUMMARY")
    print(f"{'='*80}")
    
    # Drone stats
    if 'drone_augmentation' in stats and stats['drone_augmentation']:
        drone_stats = stats['drone_augmentation']
        print(f"\nDrone Augmentation (Class 1):")
        print(f"  Total generated: {drone_stats.get('total_generated', 0)}")
        print(f"  Errors: {drone_stats.get('errors', 0)}")
        
        if 'categories' in drone_stats:
            for cat_name, cat_stats in drone_stats['categories'].items():
                print(f"\n    {cat_name}:")
                print(f"      Generated: {cat_stats['generated']} / {cat_stats['target']}")
                if 'avg_snr_db' in cat_stats:
                    print(f"      Avg SNR achieved: {cat_stats['avg_snr_db']:.2f} dB")
    
    # No-drone stats
    if 'no_drone_augmentation' in stats and stats['no_drone_augmentation']:
        no_drone_stats = stats['no_drone_augmentation']
        print(f"\nNo-Drone Augmentation (Class 0):")
        print(f"  Total generated: {no_drone_stats.get('total_generated', 0)}")
        print(f"  Errors: {no_drone_stats.get('errors', 0)}")
        
        if 'categories' in no_drone_stats:
            for cat_name, cat_stats in no_drone_stats['categories'].items():
                print(f"\n    {cat_name}:")
                print(f"      Generated: {cat_stats['generated']} / {cat_stats['target']}")
    
    # Overall stats
    print(f"\n{'─'*80}")
    print(f"TOTAL Generated: {stats.get('total_generated', 0)}")
    print(f"TOTAL Errors: {stats.get('total_errors', 0)}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced dataset augmentation with support for both drone and no-drone classes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate augmented dataset with default config
  python augment_dataset_v2.py
  
  # Use custom config
  python augment_dataset_v2.py --config augment_config_v2.json
  
  # Dry run (see what would be generated without creating files)
  python augment_dataset_v2.py --dry-run
  
  # Use legacy config (drone-only augmentation)
  python augment_dataset_v2.py --config augment_config.json
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='augment_config_v2.json',
        help='Path to augmentation configuration JSON file (default: augment_config_v2.json)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without creating files'
    )
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    
    # Load configuration
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return 1
    
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Generate augmented samples
    stats = generate_augmented_samples(config, script_dir, dry_run=args.dry_run)
    
    # Print summary
    print_summary(stats)
    
    if args.dry_run:
        print("[INFO] This was a dry run. Run without --dry-run to actually generate files.\n")
    else:
        print("[SUCCESS] Augmentation complete!\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
