#!/usr/bin/env python3
"""
Archive current models and results, then cleanup for fresh training
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path

def main():
    print("=" * 80)
    print("ARCHIVING CURRENT MODELS AND RESULTS")
    print("=" * 80)
    
    # Create archive directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"archive_easy_config_{timestamp}"
    archive_dir = Path(archive_name)
    archive_dir.mkdir(exist_ok=True)
    
    print(f"\n[1/5] Creating archive directory: {archive_name}/")
    
    # Archive saved models
    print("\n[2/5] Archiving saved models...")
    models_src = Path("saved_models")
    if models_src.exists():
        models_dst = archive_dir / "saved_models"
        shutil.copytree(models_src, models_dst)
        print(f"  ✓ Copied: saved_models/ -> {models_dst}/")
    else:
        print("  ⚠ No saved_models/ directory found")
    
    # Archive results
    print("\n[3/5] Archiving results...")
    results_src = Path("results")
    if results_src.exists():
        results_dst = archive_dir / "results"
        shutil.copytree(results_src, results_dst)
        print(f"  ✓ Copied: results/ -> {results_dst}/")
    else:
        print("  ⚠ No results/ directory found")
    
    # Archive visualization outputs
    print("\n[4/5] Archiving visualizations...")
    viz_src = Path("../6 - Visualization/outputs")
    if viz_src.exists():
        viz_dst = archive_dir / "visualization_outputs"
        shutil.copytree(viz_src, viz_dst)
        print(f"  ✓ Copied: 6 - Visualization/outputs/ -> {viz_dst}/")
    else:
        print("  ⚠ No visualization outputs found")
    
    # Archive current config
    print("\n[5/5] Archiving configuration...")
    config_src = Path("augment_config_v2.json")
    if config_src.exists():
        config_dst = archive_dir / "augment_config_v2_EASY.json"
        shutil.copy2(config_src, config_dst)
        print(f"  ✓ Copied: augment_config_v2.json -> {config_dst}")
    
    # Save archive metadata
    metadata = {
        "archive_date": timestamp,
        "archive_name": archive_name,
        "description": "Easy configuration archive - Models achieved 100% on augmented data",
        "config_summary": {
            "snr_range": "-15dB to +5dB",
            "max_noise_sources": 3,
            "model_performance": {
                "CNN": "98.90%",
                "RNN": "98.92%",
                "CRNN": "98.98%"
            }
        }
    }
    
    metadata_file = archive_dir / "archive_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved: {metadata_file}")
    
    print("\n" + "=" * 80)
    print("CLEANUP")
    print("=" * 80)
    
    # Cleanup for fresh training
    print("\n[1/7] Removing dataset_test...")
    dataset_test = Path("dataset_test")
    if dataset_test.exists():
        shutil.rmtree(dataset_test)
        print(f"  ✓ Deleted: {dataset_test}/")
    else:
        print(f"  ⚠ Not found: {dataset_test}/")
    
    print("\n[2/7] Removing dataset_train...")
    dataset_train = Path("dataset_train")
    if dataset_train.exists():
        shutil.rmtree(dataset_train)
        print(f"  ✓ Deleted: {dataset_train}/")
    else:
        print(f"  ⚠ Not found: {dataset_train}/")
    
    print("\n[3/7] Removing dataset_val...")
    dataset_val = Path("dataset_val")
    if dataset_val.exists():
        shutil.rmtree(dataset_val)
        print(f"  ✓ Deleted: {dataset_val}/")
    else:
        print(f"  ⚠ Not found: {dataset_val}/")
    
    print("\n[4/7] Removing old augmented dataset...")
    aug_dataset = Path("dataset_augmented")
    if aug_dataset.exists():
        shutil.rmtree(aug_dataset)
        print(f"  ✓ Deleted: {aug_dataset}/")
    else:
        print(f"  ⚠ Not found: {aug_dataset}/")
    
    print("\n[5/7] Removing old combined dataset...")
    combined_dataset = Path("dataset_combined")
    if combined_dataset.exists():
        shutil.rmtree(combined_dataset)
        print(f"  ✓ Deleted: {combined_dataset}/")
    else:
        print(f"  ⚠ Not found: {combined_dataset}/")
    
    print("\n[6/7] Removing old extracted features...")
    features = Path("extracted_features")
    if features.exists():
        shutil.rmtree(features)
        print(f"  ✓ Deleted: {features}/")
    else:
        print(f"  ⚠ Not found: {features}/")
    
    print("\n[7/7] Removing old models and results...")
    if models_src.exists():
        shutil.rmtree(models_src)
        print(f"  ✓ Deleted: {models_src}/")
    else:
        print(f"  ⚠ Not found: {models_src}/")
    if results_src.exists():
        shutil.rmtree(results_src)
        print(f"  ✓ Deleted: {results_src}/")
    else:
        print(f"  ⚠ Not found: {results_src}/")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Archive complete and workspace cleaned!")
    print("=" * 80)
    print(f"\nArchive location: {archive_dir.absolute()}/")
    print("\nNew configuration ready:")
    print("  • SNR range: -32dB to 0dB (500m to 50m)")
    print("  • Max noise sources: 5")
    print("  • Expected to be MUCH harder for models")
    print("\nNext steps:")
    print("  1. Run: python master_setup_v2.py")
    print("  2. Train models with new challenging dataset")
    print("=" * 80)

if __name__ == "__main__":
    main()
