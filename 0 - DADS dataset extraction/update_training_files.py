"""
Helper script to update compatibility files after feature extraction.
Run this after extracting Mel and MFCC features.
"""

import os
import shutil
from pathlib import Path

def update_files():
    script_dir = Path(__file__).parent
    features_dir = script_dir / "extracted_features"
    
    updates = {
        'mel_pitch_shift_9.0.json': 'mel_data.json',
        'mfcc_pitch_shift_9.0.json': 'mfcc_data.json',
    }
    
    print("Updating training compatibility files...")
    for target, source in updates.items():
        source_path = features_dir / source
        target_path = features_dir / target
        
        if source_path.exists():
            shutil.copy2(source_path, target_path)
            print(f"  ✓ Updated {target}")
        else:
            print(f"  ✗ Source not found: {source}")
    
    print("\nDone! You can now run the training scripts.")

if __name__ == "__main__":
    update_files()
