import os
import librosa
import math
import json
import warnings
import numpy as np
import argparse
from pathlib import Path

# Import centralized configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Suppress the librosa warning about n_fft being too large
warnings.filterwarnings('ignore', message='n_fft=.*is too large for input signal of length=.*')


def spec_augment(mel_spec, time_mask_width=10, freq_mask_width=8, num_masks=2):
    """
    SpecAugment: Mask random time/frequency bands to improve robustness.
    
    Args:
        mel_spec: Mel spectrogram (time_steps, n_mels)
        time_mask_width: Maximum width of time mask in frames
        freq_mask_width: Maximum width of frequency mask in mel bins
        num_masks: Number of masks to apply for each type
    
    Returns:
        Augmented mel spectrogram
    
    Reference:
        Park et al. "SpecAugment" (2019) https://arxiv.org/abs/1904.08779
    """
    augmented = mel_spec.copy()
    time_steps, n_mels = augmented.shape
    
    # Time masking (mask temporal columns)
    for _ in range(num_masks):
        if time_steps > time_mask_width:
            t = np.random.randint(0, time_steps - time_mask_width)
            augmented[t:t+time_mask_width, :] = augmented.mean()
    
    # Frequency masking (mask mel bins)
    for _ in range(num_masks):
        if n_mels > freq_mask_width:
            f = np.random.randint(0, n_mels - freq_mask_width)
            augmented[:, f:f+freq_mask_width] = augmented.mean()
    
    return augmented

# Default audio params (may be overridden by centralized `config.py` below)
SAMPLE_RATE = config.SAMPLE_RATE
DURATION = config.DURATION
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

# Default dataset path and output path are derived from `config`.
DEFAULT_DATASET_PATH = Path(getattr(config, 'DATASET_ROOT_STR', '.'))



def save_mfcc(dataset_path, out_path, n_mels=config.MEL_N_MELS, n_fft=config.MEL_N_FFT, hop_length=config.MEL_HOP_LENGTH, num_segments=config.NUM_SEGMENTS, apply_spec_augment=False, output_format='npz'):
    # num_segments let's you chop up track into different segments to create a bigger dataset.
    # Value is changed at the bottom of the script.

    # Dictionary to store data (in-memory)
    data = {
        "mapping": [],  # Used to map labels (0 and 1) to category name (UAV and no UAV).
        "mel": [],  # Mels are the training input, labels are the target.
        "labels": []  # Features are mapped to a label (0 or 1).
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mel_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # Count total files for progress tracking
    total_files = 0
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        if dirpath != str(dataset_path):
            total_files += len(filenames)
    
    processed_files = 0
    progress_interval = max(1, total_files // 5)  # 20% intervals

    # Prepare index containers (one MEL per WAV)
    index_names = []
    index_mels = []

    # Loops through all the folders in the training audio folder.
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Ensures that we're not at the root level.
        if dirpath is not dataset_path:

            # Saves the semantic label for the mapping.
            dirpath_components = dirpath.split(os.sep)  # class/background => ["class", "background"]
            semantic_label = dirpath_components[-1]  # considering only the last value
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            # Processes all the audio files for a specific class.
            for f in filenames:
                # Progress tracking (20% intervals)
                processed_files += 1
                if processed_files % progress_interval == 0 or processed_files == total_files:
                    pct = (processed_files / total_files * 100) if total_files > 0 else 0
                    print(f"Progress: {processed_files}/{total_files} ({pct:.0f}%)", flush=True)

                # Loads audio file.
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=config.MEL_DURATION)

                # Precompute canonical mel for this WAV (non-augmented) by taking the
                # central segment if num_segments > 1, or the only segment when ==1.
                # We'll compute the mel for the full signal and extract the central
                # segment's mel frame-wise to ensure shape consistency.
                # Compute full mel spec for canonical storage (n_mels, frames)
                full_mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length)
                # Use same reference as training/inference (ref=np.max) to ensure consistent dB scaling
                full_db = librosa.power_to_db(full_mel, ref=np.max)
                # Normalize/truncate/pad canonical mel to MEL_TIME_FRAMES
                if full_db.shape[1] < config.MEL_TIME_FRAMES:
                    pad_width = config.MEL_TIME_FRAMES - full_db.shape[1]
                    canonical_mel = np.pad(full_db, ((0,0),(0,pad_width)), mode='constant', constant_values=(config.MEL_PAD_VALUE,))
                else:
                    canonical_mel = full_db[:, :config.MEL_TIME_FRAMES]
                # Store index entry (names and mel as list)
                index_names.append(f)
                index_mels.append(canonical_mel.tolist())

                # Process segments, extracting mels and storing data to JSON_PATH.
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment  # s=0 --> num_samples_per_segment

                    # Extract segment and pad if necessary
                    segment = signal[start_sample:finish_sample]
                    
                    # Pad segment with zeros if it's shorter than expected
                    if len(segment) < num_samples_per_segment:
                        segment = np.pad(segment, (0, num_samples_per_segment - len(segment)), mode='constant')

                    mel_spec = librosa.feature.melspectrogram(y=segment,
                                                         sr=sr,
                                                         n_fft=n_fft,
                                                         n_mels=n_mels,
                                                         hop_length=hop_length)
                    # Use same dB reference as trainers/evaluators
                    db_mel = librosa.power_to_db(mel_spec, ref=np.max)
                    # Keep orientation (n_mels, time) to match training/evaluation code
                    mel = db_mel
                    
                    # Apply SpecAugment with 50% probability (configurable via global)
                    if apply_spec_augment:
                        # When enabled, apply SpecAugment with 50% probability.
                        # spec_augment expects (time_steps, n_mels), so transpose, augment, then transpose back.
                        if np.random.rand() < 0.5:
                            aug = spec_augment(mel.T,
                                               time_mask_width=10,
                                               freq_mask_width=8,
                                               num_masks=2)
                            mel = aug.T

                    # Stores mels for segment, if it has the expected length (time axis == expected frames).
                    if mel.shape[1] == expected_num_mel_vectors_per_segment:
                        data["mel"].append(mel)
                        data["labels"].append(i - 1)

    # Convert lists to numpy arrays for efficient storage
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    mels_array = np.array(data["mel"], dtype=np.float32)
    labels_array = np.array(data["labels"], dtype=np.int32)
    mapping_array = np.array(data["mapping"])
    
    # Write NPZ format (default, 5x smaller, 10x faster loading)
    if output_format == 'npz':
        npz_path = out_path.with_suffix('.npz')
        np.savez_compressed(
            str(npz_path),
            mels=mels_array,
            labels=labels_array,
            mapping=mapping_array
        )
        print(f"\nSaved NPZ: {npz_path} ({npz_path.stat().st_size / (1024**2):.2f} MB)", flush=True)
    
    # Write legacy JSON format (optional, for backward compatibility)
    if output_format == 'json' or output_format == 'both':
        json_data = {
            "mapping": data["mapping"],
            "mel": [m.tolist() for m in data["mel"]],
            "labels": data["labels"]
        }
        json_path = out_path.with_suffix('.json')
        with open(json_path, "w") as fp:
            json.dump(json_data, fp, indent=4)
        print(f"Saved JSON: {json_path} ({json_path.stat().st_size / (1024**2):.2f} MB)", flush=True)


if __name__ == "__main__":
    # Par
    parser = argparse.ArgumentParser(description='Extract Mel spectrograms with optional SpecAugment')
    parser.add_argument('--spec_augment', choices=['auto','on','off'], default='auto',
                       help='Apply SpecAugment: auto=enable for train per config, on=force, off=disable')
    parser.add_argument('--split', choices=['train','val','test'], default='train')
    parser.add_argument('--num_segments', type=int, default=None,
                       help='Number of segments per audio file (default: 1)')
    parser.add_argument('--output-format', choices=['npz', 'json', 'both'], default='npz',
                       help='Output format: npz (default, fast), json (legacy), both (compatibility)')
    args = parser.parse_args()
    
    # Determine split and default behavior
    split = args.split
    # Default spec augment enabled for training when configured
    if args.spec_augment == 'on':
        apply_spec = True
    elif args.spec_augment == 'off':
        apply_spec = False
    else:  # auto
        if split == 'train' and getattr(config, 'SPEC_AUGMENT_BY_DEFAULT_FOR_TRAIN', True):
            apply_spec = True
        else:
            apply_spec = False
    
    # (Display will occur after we resolve dataset/output paths below)
    
    # Use centralized config defaults where available
    n_mels = getattr(config, 'MEL_N_MELS', 90)
    n_fft = getattr(config, 'MEL_N_FFT', 2048)
    hop = getattr(config, 'MEL_HOP_LENGTH', 512)
    segments = args.num_segments if args.num_segments is not None else getattr(config, 'NUM_SEGMENTS', 1)

    # Resolve dataset path supporting two common layouts:
    # 1) DATASET_ROOT/{train,val,test} (e.g. dataset_combined/train)
    # 2) sibling folders named dataset_train, dataset_val, dataset_test
    dataset_root = Path(config.DATASET_ROOT) if hasattr(config, 'DATASET_ROOT') else Path(DEFAULT_DATASET_PATH)
    parent_dir = dataset_root.parent
    candidate1 = parent_dir / f"dataset_{split}"

    if candidate1.exists():
        dataset_path = candidate1
    else:
        raise FileNotFoundError(f"Could not find dataset split for '{split}'. Tried: '{candidate1}'")

    # Determine output path per split
    out_path = Path(config.EXTRACTED_FEATURES_DIR) / f"mel_{split}.json"

    # Display configuration
    print("\n" + "="*60)
    print("MEL SPECTROGRAM FEATURE EXTRACTION")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {out_path}")
    print(f"Output Format: {args.output_format.upper()}")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Duration: {DURATION} seconds")
    print(f"Segments: {segments}")
    print(f"SpecAugment: {'ENABLED (50% probability)' if apply_spec else 'DISABLED'}")
    print("="*60 + "\n")

    save_mfcc(dataset_path, out_path, n_mels=n_mels, n_fft=n_fft, hop_length=hop, num_segments=segments, apply_spec_augment=apply_spec, output_format=args.output_format)
    
    print("\n" + "="*60)
    print("Feature extraction complete!")
    print(f"Output: {out_path}")
    print("="*60)