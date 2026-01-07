"""
Modern Audio Examples Generator
Génère des exemples audio représentatifs avec visualisations à partir du dataset.

Features:
- Utilise config.py pour tous les chemins
- Exemples de drones (différentes distances) et ambients (différents types)
- Waveforms et spectrogrammes
- Page HTML avec lecteurs audio
- Pas de fallback, tout depuis config
"""

import sys
from pathlib import Path
import shutil
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random

# Import project config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from tools import plot_utils

# Apply plotting style
plot_utils.set_style()

# Output directory
OUTPUT_DIR = plot_utils.get_output_dir(__file__) / 'audio_examples'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_example_files():
    """
    Trouve des fichiers exemples représentatifs dans le dataset de test.
    Retourne dict avec catégories clés.
    """
    test_dir = config.DATASET_TEST_DIR
    
    examples = {
        'drone_close': [],      # 100m, 200m
        'drone_medium': [],     # 350m, 500m
        'drone_far': [],        # 600m, 700m, 850m, 1000m
        'drone_original': [],   # dads_1_*.wav
        'ambient_complex': [],
        'ambient_extreme': [],
        'ambient_moderate': [],
        'ambient_very': [],
        'ambient_original': []  # dads_0_*.wav
    }
    
    if not test_dir.exists():
        print(f"[ERROR] Test directory not found: {test_dir}")
        return examples
    
    # Parcourir class 0 et class 1
    for class_dir in [test_dir / '0', test_dir / '1']:
        if not class_dir.exists():
            continue
        
        for wav_file in class_dir.glob('*.wav'):
            fname = wav_file.name.lower()
            
            # Drones
            if '100m' in fname or '200m' in fname:
                examples['drone_close'].append(wav_file)
            elif '350m' in fname or '500m' in fname:
                examples['drone_medium'].append(wav_file)
            elif '600m' in fname or '700m' in fname or '850m' in fname or '1000m' in fname:
                examples['drone_far'].append(wav_file)
            elif 'dads_1_' in fname:
                examples['drone_original'].append(wav_file)
            
            # Ambients
            elif 'ambient_complex' in fname or 'ambient complex' in fname:
                examples['ambient_complex'].append(wav_file)
            elif 'ambient_extreme' in fname or 'ambient extreme' in fname:
                examples['ambient_extreme'].append(wav_file)
            elif 'ambient_moderate' in fname or 'ambient moderate' in fname:
                examples['ambient_moderate'].append(wav_file)
            elif 'ambient_very' in fname or 'ambient very' in fname:
                examples['ambient_very'].append(wav_file)
            elif 'dads_0_' in fname:
                examples['ambient_original'].append(wav_file)
    
    return examples


def generate_visualizations(audio_path, output_dir):
    """
    Génère waveform et spectrogramme pour un fichier audio.
    """
    sr = getattr(config, 'SAMPLE_RATE', 22050)
    
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        print(f"  ✗ Error loading {audio_path.name}: {e}")
        return False
    
    name = audio_path.stem
    
    # Figure avec 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title(f'Waveform - {name}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Spectrogramme (Mel)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax2, cmap='viridis')
    ax2.set_title(f'Mel Spectrogram - {name}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    fig.colorbar(img, ax=ax2, format='%+2.0f dB')
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = output_dir / f'{name}_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def copy_audio_file(src_path, output_dir):
    """Copie le fichier audio dans le dossier de sortie."""
    dst_path = output_dir / src_path.name
    shutil.copy2(src_path, dst_path)
    return dst_path.name


def generate_html_page(selected_files, output_dir):
    """
    Génère une page HTML avec lecteurs audio et visualisations.
    """
    html_content = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Examples - Acoustic UAV Identification</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 40px;
            border-left: 5px solid #e74c3c;
            padding-left: 15px;
        }
        .example-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .example-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #2980b9;
            margin-bottom: 10px;
        }
        audio {
            width: 100%;
            margin: 10px 0;
        }
        img {
            width: 100%;
            border-radius: 4px;
            margin-top: 15px;
        }
        .category {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 6px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <h1>🎵 Audio Examples - Acoustic UAV Identification</h1>
    <p style="text-align: center; color: #7f8c8d; font-size: 0.9em;">
        Exemples représentatifs du dataset de test avec visualisations
    </p>
"""
    
    for category, files in selected_files.items():
        if not files:
            continue
        
        # Titre de catégorie
        category_display = category.replace('_', ' ').title()
        html_content += f'\n    <h2>{category_display}</h2>\n'
        
        for audio_file, vis_file in files:
            html_content += f'''
    <div class="example-container">
        <div class="example-title">{audio_file}</div>
        <audio controls>
            <source src="{audio_file}" type="audio/wav">
            Votre navigateur ne supporte pas l'élément audio.
        </audio>
        <img src="{vis_file}" alt="Visualization for {audio_file}">
    </div>
'''
    
    html_content += """
</body>
</html>
"""
    
    html_path = output_dir / 'index.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"  ✓ HTML page: {html_path}")


def main():
    """Point d'entrée principal."""
    print("\n" + "="*80)
    print("AUDIO EXAMPLES GENERATION")
    print("="*80 + "\n")
    
    # Trouver les fichiers exemples
    print("[1/4] Finding example files...")
    examples = find_example_files()
    
    # Sélectionner 1-2 fichiers par catégorie
    selected_files = {}
    total_selected = 0
    
    for category, files in examples.items():
        if not files:
            print(f"  ⚠ No files found for {category}")
            continue
        
        # Prendre 1-2 exemples aléatoires
        n_samples = min(2, len(files))
        selected = random.sample(files, n_samples)
        selected_files[category] = []
        
        print(f"  ✓ {category}: {len(files)} files, selected {n_samples}")
        total_selected += n_samples
    
    if total_selected == 0:
        print("\n[ERROR] No example files found!")
        return
    
    print(f"\n[2/4] Generating visualizations for {total_selected} files...")
    
    # Générer visualisations et copier fichiers audio
    for category, files in selected_files.items():
        for audio_path in examples[category][:2]:  # Max 2 par catégorie
            vis_success = generate_visualizations(audio_path, OUTPUT_DIR)
            if vis_success:
                audio_filename = copy_audio_file(audio_path, OUTPUT_DIR)
                vis_filename = f'{audio_path.stem}_visualization.png'
                selected_files[category].append((audio_filename, vis_filename))
                print(f"  ✓ Processed: {audio_path.name}")
    
    print(f"\n[3/4] Generating HTML page...")
    generate_html_page(selected_files, OUTPUT_DIR)
    
    print(f"\n[4/4] Summary...")
    print(f"  Total examples: {sum(len(v) for v in selected_files.values())}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Open: file://{OUTPUT_DIR / 'index.html'}")
    
    print("\n" + "="*80)
    print("✓ Audio examples generation complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
