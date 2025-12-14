import os
import librosa
import math
import json
import warnings
import numpy as np
import argparse

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

# Import configuration from centralized config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    import config
    DATASET_PATH = config.DATASET_ROOT_STR
    JSON_PATH = config.MEL_TRAIN_PATH_STR
except ImportError:
    # Fallback to manual paths if config doesn't exist
    DATASET_PATH = "..."  # Path of folder with training audios.
    JSON_PATH = ".../mel_data.json"  # Location and file name to save feature extracted data.

SAMPLE_RATE = 22050  # Sample rate in Hz.
DURATION = 10  # Length of audio files fed. Measured in seconds.
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, n_mels=90, n_fft=2048, hop_length=512, num_segments=5):
    # num_segments let's you chop up track into different segments to create a bigger dataset.
    # Value is changed at the bottom of the script.

    # Dictionary to store data into JSON_PATH
    data = {
        "mapping": [],  # Used to map labels (0 and 1) to category name (UAV and no UAV).
        "mel": [],  # Mels are the training input, labels are the target.
        "labels": []  # Features are mapped to a label (0 or 1).
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mel_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

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

                # Loads audio file.
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # Process segments, extracting mels and storing data to JSON_PATH.
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment  # s=0 --> num_samples_per_segment

                    # Extract segment and pad if necessary
                    segment = signal[start_sample:finish_sample]
                    
                    # Pad segment with zeros if it's shorter than expected
                    if len(segment) < num_samples_per_segment:
                        segment = np.pad(segment, (0, num_samples_per_segment - len(segment)), mode='constant')

                    mel = librosa.feature.melspectrogram(y=segment,
                                                         sr=sr,
                                                         n_fft=n_fft,
                                                         n_mels=n_mels,
                                                         hop_length=hop_length)
                    db_mel = librosa.power_to_db(mel)
                    mel = db_mel.T
                    
                    # Apply SpecAugment with 50% probability (configurable via global)
                    if hasattr(save_mfcc, 'apply_spec_augment') and save_mfcc.apply_spec_augment:
                        if np.random.rand() < 0.5:
                            mel = spec_augment(mel, 
                                             time_mask_width=10, 
                                             freq_mask_width=8, 
                                             num_masks=2)

                    # Stores mels for segment, if it has the expected length.
                    if len(mel) == expected_num_mel_vectors_per_segment:
                        data["mel"].append(mel.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, s + 1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Extract Mel spectrograms with optional SpecAugment')
    parser.add_argument('--spec_augment', action='store_true', 
                       help='Apply SpecAugment (time/frequency masking) to 50%% of samples')
    parser.add_argument('--num_segments', type=int, default=10,
                       help='Number of segments per audio file (default: 10)')
    args = parser.parse_args()
    
    # Configure SpecAugment
    save_mfcc.apply_spec_augment = args.spec_augment
    
    # Display configuration
    print("\n" + "="*60)
    print("🎵 MEL SPECTROGRAM FEATURE EXTRACTION")
    print("="*60)
    print(f"📂 Dataset: {DATASET_PATH}")
    print(f"💾 Output: {JSON_PATH}")
    print(f"🔊 Sample Rate: {SAMPLE_RATE} Hz")
    print(f"⏱️  Duration: {DURATION} seconds")
    print(f"🔪 Segments: {args.num_segments}")
    print(f"🎭 SpecAugment: {'✅ ENABLED (50% probability)' if args.spec_augment else '❌ DISABLED'}")
    print("="*60 + "\n")
    
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=args.num_segments)
    
    print("\n" + "="*60)
    print("✅ Feature extraction complete!")
    print("="*60)
