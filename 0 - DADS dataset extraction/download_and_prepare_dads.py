"""
Script to download and prepare the DADS (Drone Audio Detection Samples) dataset from Hugging Face.
This script converts the dataset to the format expected by the Mel/MFCC preprocessing scripts.

DADS dataset: https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples
- 180,320 audio files total
- Labels: 0 (no drone), 1 (drone)
- Original format: 16 kHz, mono, WAV PCM 16-bit
- Variable duration (0.5s to several minutes)

Output format (for compatibility with existing scripts):
- Resampled to 22050 Hz (project standard)
- Organized in folders by label: output_dir/0/ and output_dir/1/
- WAV format (float32)

NOTE: Paths are managed by the centralized config.py at the project root.
"""

import os
import argparse
from datasets import load_dataset
import librosa
import soundfile as sf
from tqdm import tqdm
import config
from tools.audio_utils import ensure_duration

# Target sample rate for the project (matches Mel_Preprocess_and_Feature_Extract.py).
TARGET_SAMPLE_RATE = config.SAMPLE_RATE


def download_and_prepare_dads(output_dir, max_samples=None, max_per_class=None, verbose=True):
    """
    Download DADS dataset from Hugging Face and prepare it for the project.
    
    Args:
        output_dir (str): Root directory where to save the organized dataset.
        max_samples (int, optional): Maximum total number of samples to extract (across all classes).
        max_per_class (int, optional): Maximum number of samples per class (0 and 1).
        verbose (bool): Print progress information.
    
    Returns:
        dict: Statistics about the extraction (counts per class, skipped files, etc.)
    """
    
    if verbose:
        print("=" * 80)
        print("DADS Dataset Extraction")
        print("=" * 80)
        print(f"Output directory: {output_dir}")
        print(f"Target sample rate: {TARGET_SAMPLE_RATE} Hz")
        if max_samples:
            print(f"Max total samples: {max_samples}")
        if max_per_class:
            print(f"Max per class: {max_per_class}")
        print("=" * 80)
        print("\nLoading dataset from Hugging Face...")
    
    # Load dataset from Hugging Face with streaming disabled to download files
    # Disable audio decoding to avoid torchcodec dependency - we'll load files manually
    from datasets import Audio
    ds = load_dataset("geronimobasso/drone-audio-detection-samples", split="train", streaming=False)
    
    # Cast audio column to not decode automatically
    ds = ds.cast_column("audio", Audio(decode=False))
    
    if verbose:
        print(f"✓ Dataset loaded: {len(ds)} total samples")
    
    # Create output directory structure.
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics tracking.
    stats = {
        "total_processed": 0,
        "total_saved": 0,
        "skipped": 0,
        "errors": 0,
        "per_class": {}
    }
    
    # Count samples per class for progress tracking.
    if verbose:
        print("\nAnalyzing dataset distribution...")
    
    class_counts = {}
    for ex in ds:
        label = ex.get("label", 0)
        class_counts[label] = class_counts.get(label, 0) + 1
    
    if verbose:
        print(f"✓ Class distribution:")
        for label, count in sorted(class_counts.items()):
            print(f"  - Label {label}: {count:,} samples")
    
    # Initialize per-class directories and counters.
    for label in class_counts.keys():
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        stats["per_class"][label] = {
            "saved": 0,
            "skipped": 0,
            "errors": 0
        }
    
    # Process samples with progress bar.
    if verbose:
        print("\nProcessing audio files...")
        iterator = tqdm(enumerate(ds), desc="Extracting", unit="file", total=len(ds))
    else:
        iterator = enumerate(ds)
    
    for idx, ex in iterator:
        # Check global limit.
        if max_samples and stats["total_saved"] >= max_samples:
            if verbose:
                print(f"\n✓ Reached max_samples limit ({max_samples}). Stopping.")
            break
        
        try:
            # Extract label from dataset example
            label = ex.get("label", 0)
            
            # Check per-class limit.
            if max_per_class and stats["per_class"][label]["saved"] >= max_per_class:
                stats["per_class"][label]["skipped"] += 1
                stats["skipped"] += 1
                continue
            
            # Get audio data - when decode=False, HuggingFace returns bytes and path
            audio_data = ex.get("audio")
            if audio_data is None:
                stats["per_class"][label]["errors"] += 1
                stats["errors"] += 1
                continue
            
            # audio_data is a dict with 'bytes' and 'path' when decode=False
            if isinstance(audio_data, dict):
                audio_bytes = audio_data.get("bytes")
                filename = audio_data.get("path", f"audio_{idx}.wav")
            else:
                stats["per_class"][label]["errors"] += 1
                stats["errors"] += 1
                continue
            
            if not audio_bytes:
                stats["per_class"][label]["errors"] += 1
                stats["errors"] += 1
                continue
            
            # Load audio from bytes using soundfile
            import io
            audio_array, original_sr = sf.read(io.BytesIO(audio_bytes))

            # Ensure mono: if multi-channel, convert to mono by averaging channels
            if hasattr(audio_array, 'ndim') and audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Resample to target sample rate if needed.
            if original_sr != TARGET_SAMPLE_RATE:
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=original_sr,
                    target_sr=TARGET_SAMPLE_RATE
                )
            
            # Create output path using original filename if available
            label_dir = os.path.join(output_dir, str(label))
            output_filename = f"dads_{label}_{stats['per_class'][label]['saved']:06d}.wav"
            output_path = os.path.join(label_dir, output_filename)
            
            # Ensure float32 dtype and resample if necessary
            audio_array = audio_array.astype('float32')

            # Enforce exact target duration by looping/trimming with crossfade
            audio_array = ensure_duration(audio_array, TARGET_SAMPLE_RATE, float(config.AUDIO_DURATION_S), crossfade_duration=0.1)

            # Save as WAV file using project-configured subtype.
            try:
                subtype = getattr(config, 'AUDIO_WAV_SUBTYPE', 'FLOAT')
            except Exception:
                subtype = 'FLOAT'

            sf.write(output_path, audio_array, TARGET_SAMPLE_RATE, subtype=subtype)
            
            # Update statistics.
            stats["per_class"][label]["saved"] += 1
            stats["total_saved"] += 1
            stats["total_processed"] += 1
            
        except KeyboardInterrupt:
            if verbose:
                print("\n\n⚠ Interrupted by user. Saving progress...")
            break
        except Exception as e:
            # Log error and continue.
            stats["per_class"].get(label, stats["per_class"][0])["errors"] += 1
            stats["errors"] += 1
            if verbose and stats["errors"] <= 10:  # Only show first 10 errors
                print(f"\nError processing sample {idx}: {str(e)}")
            continue
    
    # Print final statistics.
    if verbose:
        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"Total processed: {stats['total_processed']:,}")
        print(f"Total saved: {stats['total_saved']:,}")
        print(f"Skipped (limits): {stats['skipped']:,}")
        print(f"Errors: {stats['errors']:,}")
        print("\nPer-class statistics:")
        for label, class_stats in sorted(stats["per_class"].items()):
            print(f"  Label {label}:")
            print(f"    - Saved: {class_stats['saved']:,}")
            print(f"    - Skipped: {class_stats['skipped']:,}")
            print(f"    - Errors: {class_stats['errors']:,}")
        print("=" * 80)
        print(f"\n✓ Dataset ready at: {os.path.abspath(output_dir)}")
        
        # Create feature extraction directory
        base_dir = os.path.dirname(os.path.abspath(output_dir))
        json_dir = os.path.join(base_dir, "extracted_features")
        os.makedirs(json_dir, exist_ok=True)
        
        print(f"✓ Feature extraction directory created at: {json_dir}")
        
        print(f"\nNext steps:")
        print(f"  1. Run the preprocessing scripts to extract features (Mel/MFCC)")
        print(f"     - They will use paths from the centralized config.py")
        print(f"  2. Train your models with the extracted features")
        print("=" * 80)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare DADS dataset from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all samples (warning: ~180k files, may take hours)
  python download_and_prepare_dads.py --output dataset_full
  
  # Extract limited number for quick testing
  python download_and_prepare_dads.py --output dataset_test --max-samples 1000
  
  # Extract balanced dataset (1000 per class)
  python download_and_prepare_dads.py --output dataset_balanced --max-per-class 1000
  
  # Extract small subset for development
  python download_and_prepare_dads.py --output dataset_dev --max-per-class 100
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="dataset",
        help="Output directory for the organized dataset (default: 'dataset')"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum total number of samples to extract (default: no limit)"
    )
    
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Maximum number of samples per class (default: no limit)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Run extraction.
    stats = download_and_prepare_dads(
        output_dir=args.output,
        max_samples=args.max_samples,
        max_per_class=args.max_per_class,
        verbose=not args.quiet
    )
    
    return stats


if __name__ == "__main__":
    main()
