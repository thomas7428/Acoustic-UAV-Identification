import os
import librosa
import math
import json
import warnings
import numpy as np
from pathlib import Path
import argparse
import sys

# Suppress the librosa warning about n_fft being too large
warnings.filterwarnings('ignore', message='n_fft=.*is too large for input signal of length=.*')

# Import centralized configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

SAMPLE_RATE = config.SAMPLE_RATE  # Sample rate in Hz.
DURATION = config.DURATION  # Length of audio files fed. Measured in seconds. MUST MATCH TRAINING!
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Default dataset root
DEFAULT_DATASET_PATH = Path(getattr(config, 'DATASET_ROOT', '.'))

def save_mfcc(dataset_path, out_path, n_mfcc=config.MFCC_N_MFCC, n_fft=config.MEL_N_FFT, hop_length=config.MEL_HOP_LENGTH, num_segments=config.NUM_SEGMENTS):
    # num_segments let's you chop up track into different segments to create a bigger dataset.
    # Value is changed at the bottom of the script.

    # Dictionary to store data into JSON_PATH
    data = {
        "mapping": [],  # Used to map labels (0 and 1) to category name (UAV and no UAV).
        "mfcc": [],  # MFCCs are the training input, labels are the target.
        "labels": []  # Features are mapped to a label (0 or 1).
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) # 1.2 -> 2

    # Prepare index containers (one MFCC per WAV)
    index_names = []
    index_mfccs = []

    # Loops through all the folders in the training audio folder.
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Ensures that we're not at the root level.
        if dirpath is not dataset_path:

            # Saves the semantic label for the mapping.
            dirpath_components = dirpath.split(os.sep)     # class/background => ["class", "background"]
            semantic_label = dirpath_components[-1]     # considering only the last value
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            # Processes all the audio files for a specific class.
            for f in filenames:

                # Loads audio file.
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=config.MEL_DURATION)

                # Precompute canonical MFCC for this WAV (non-augmented)
                full_mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
                # full_mfcc shape is (n_mfcc, frames)
                if full_mfcc.shape[1] < config.MEL_TIME_FRAMES:
                    pad_w = config.MEL_TIME_FRAMES - full_mfcc.shape[1]
                    canonical = np.pad(full_mfcc, ((0,0),(0,pad_w)), mode='constant', constant_values=(config.MEL_PAD_VALUE,))
                else:
                    canonical = full_mfcc[:, :config.MEL_TIME_FRAMES]
                index_names.append(f)
                index_mfccs.append(canonical.tolist())

                # Process segments, extracting mfccs and storing data to JSON_PATH.
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s # s=0 --> 0
                    finish_sample = start_sample + num_samples_per_segment # s=0 --> num_samples_per_segment

                    # Extract segment and pad if necessary
                    segment = signal[start_sample:finish_sample]
                    
                    # Pad segment with zeros if it's shorter than expected
                    if len(segment) < num_samples_per_segment:
                        segment = np.pad(segment, (0, num_samples_per_segment - len(segment)), mode='constant')

                    mfcc = librosa.feature.mfcc(y=segment,
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # Stores mfccs for segment, if it has the expected length.
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s+1))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(data, fp, indent=4)

    # Write index file: one MFCC per WAV
    index_path = Path(config.EXTRACTED_FEATURES_DIR) / f"mfcc_{split}_index.json"
    with open(index_path, 'w') as idxf:
        json.dump({"names": index_names, "mfccs": index_mfccs}, idxf)

    if getattr(save_mfcc, 'write_npz', False):
        npz_path = index_path.with_suffix('.npz')
        names_arr = np.array(index_names)
        mfccs_arr = np.array(index_mfccs)
        np.savez_compressed(str(npz_path), names=names_arr, mfccs=mfccs_arr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract MFCCs per split')
    parser.add_argument('--split', choices=['train','val','test'], default='train')
    parser.add_argument('--num_segments', type=int, default=None)
    parser.add_argument('--write-npz', action='store_true', dest='write_npz', help='Also write compressed .npz index')
    args = parser.parse_args()

    split = args.split
    segments = args.num_segments if args.num_segments is not None else getattr(config, 'NUM_SEGMENTS', 10)
    
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
    
    out_path = Path(config.EXTRACTED_FEATURES_DIR) / f"mfcc_{split}.json"

    # Display configuration
    print("\n" + "="*60)
    print("MFCC FEATURE EXTRACTION")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {out_path}")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Duration: {DURATION} seconds")
    print(f"Segments: {args.num_segments}")
    print("="*60 + "\n")

    # Allow optional writing of .npz index via a flag
    save_mfcc.write_npz = args.write_npz
    save_mfcc(dataset_path, out_path, num_segments=segments)
    
    print("\n" + "="*60)
    print("Feature extraction complete!")
    print(f"Output: {out_path}")
    print("="*60)