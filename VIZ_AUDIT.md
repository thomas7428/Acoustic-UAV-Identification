# Audit du Dossier Visualisation (6 - Visualization/)
**Date**: 2026-01-10

---

## 📊 Vue d'ensemble

**13 scripts Python** | ~3600 lignes total

### Scripts de Run (2 runners)
1. **run_all_visualizations.py** (151 lignes) - Runner moderne avec imports directs
2. **run_enhanced_visualizations.py** (125 lignes) - Runner subprocess avec pipeline

### Scripts Core (11 visualizers)
| Script | Lignes | Type | Statut |
|--------|--------|------|--------|
| performance_comparison.py | 640 | Core/Legacy | 🟡 **Obsolète partiel** |
| performance_comparison_best.py | 359 | Core | ✅ **Moderne** |
| generate_html_report.py | 525 | Generator | ⚠️ **À améliorer** |
| modern_threshold_calibration.py | 302 | Analysis | ✅ **OK** |
| modern_dataset_analysis.py | 300 | Analysis | ✅ **OK** |
| modern_audio_examples.py | 298 | Generator | ✅ **OK** |
| model_comparison_plots.py | 259 | Plots | ✅ **OK** |
| snr_distribution.py | 218 | Plots | ✅ **OK** |
| threshold_calibration_comparison.py | 180 | Plots | ✅ **OK** |
| select_best_results.py | 161 | Utility | ✅ **OK** |
| quick_viz.py | 76 | Launcher | ⚠️ **Dépend script obsolète** |

---

## 🔴 PROBLÈMES IDENTIFIÉS

### 1. **REDONDANCE: Deux Runners avec Approches Différentes**

#### `run_all_visualizations.py` (151 lignes)
```python
# Approche: imports directs
import modern_dataset_analysis
import modern_audio_examples
import modern_threshold_calibration
import performance_comparison

# Puis appelle:
modern_dataset_analysis.main()
performance_comparison.main()
```

**Problèmes**:
- Importe `performance_comparison` (le gros script legacy de 640 lignes)
- Mélange ancien et nouveau code
- Gestion d'erreur try/except cache les problèmes

#### `run_enhanced_visualizations.py` (125 lignes)
```python
# Approche: subprocess
pipeline = [
    ("performance_comparison_best.py", "Step 1..."),
    ("threshold_calibration_comparison.py", "Step 3..."),
    ("model_comparison_plots.py", "Step 4..."),
    # ...
    ("generate_html_report.py", "Step 8..."),
]

for script, description in pipeline:
    subprocess.run([sys.executable, script])
```

**Problèmes**:
- Utilise subprocess (plus lent, isolation excessive)
- Pipeline hardcodé
- Pas de gestion des dépendances entre étapes
- Numérotation saute Step 2 (??!)

**Recommandation**: **Fusionner en UN SEUL runner moderne**

---

### 2. **performance_comparison.py - Script Legacy Complexe (640 lignes)**

**Utilisé par**:
- `run_all_visualizations.py` (l'ancien runner)
- `quick_viz.py` (launcher de presets)

**Problèmes**:
- 640 lignes monolithiques
- Parse arguments complexes (--models, --splits, --thresholds, --all)
- Génère BEAUCOUP de PNGs (un par threshold × modèle × split)
- Approche "tout ou rien" avec flag `--all`
- **Redondant avec `performance_comparison_best.py`** qui fait la même chose mais mieux

**Code suspect**:
```python
# Ligne 589: main() avec argparse massif
parser.add_argument('--all', action='store_true')
parser.add_argument('--models', nargs='+', choices=['CNN', 'RNN', ...])
parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'])
parser.add_argument('--thresholds', nargs='+', type=float)
# ... 20+ arguments
```

**Recommandation**: **Déprécier et rediriger vers performance_comparison_best.py**

---

### 3. **quick_viz.py - Dépend du Script Obsolète**

```python
SCRIPT_PATH = Path(__file__).parent / "performance_comparison.py"  # ← Legacy!

PRESETS = {
    "all": {"args": ["--all"]},
    "cnn-test": {"args": ["--models", "CNN", "--splits", "test"]},
    # ...
}

def main():
    cmd = [sys.executable, str(SCRIPT_PATH)] + args
    subprocess.run(cmd)
```

**Problème**: Lance `performance_comparison.py` (l'ancien script de 640 lignes)

**Recommandation**: **Mettre à jour pour utiliser performance_comparison_best.py**

---

### 4. **generate_html_report.py - Approche Base64 Lourde (525 lignes)**

```python
def encode_image_to_base64(image_path):
    """Encode une image en base64 pour l'intégration dans le HTML."""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')

# Puis dans le HTML:
html_content = f"""
<img src="data:image/png;base64,{encoded_images['threshold_calibration']}" alt="...">
"""
```

**Problèmes**:
1. **Base64 gonfl e la taille** - Une image de 500 KB → 667 KB en base64
2. **HTML devient énorme** - Plusieurs MB pour un seul fichier
3. **Pas portable** - Impossible de sauvegarder les images séparément
4. **Lent à charger** - Navigateur doit décoder le base64
5. **Dur à déboguer** - Impossible de voir les images directement

**Approche moderne** (comme `modern_audio_examples.py`):
```python
# Copier les images dans un dossier
shutil.copy(image_path, output_dir / "images" / image_name)

# Dans le HTML:
<img src="./images/threshold_calibration.png" alt="...">
```

**Recommandation**: **Refactorer pour utiliser chemins relatifs au lieu de base64**

**Autres problèmes HTML**:
- CSS inline massif (180+ lignes, lignes 124-304)
- Pas responsive pour mobile
- Hardcodé au lieu d'utiliser templates
- Génération de HTML par concaténation de strings (vulnérable, illisible)

---

### 5. **Numérotation Incohérente dans Pipelines**

#### `run_enhanced_visualizations.py`:
```python
pipeline = [
    ("performance_comparison_best.py", "Step 1: Performance Analysis"),
    ("threshold_calibration_comparison.py", "Step 3: Threshold..."),  # ← Step 2 ???
    ("model_comparison_plots.py", "Step 4: Model..."),
    ("snr_distribution.py", "Step 5: SNR..."),
    ("modern_dataset_analysis.py", "Step 6: Dataset..."),
    ("modern_threshold_calibration.py", "Step 7: Modern..."),
    ("generate_html_report.py", "Step 8: Generate HTML"),
]
```

**Step 2 est manquant!** Probablement supprimé sans renommer les suivants.

**Recommandation**: **Retirer les numéros ou les corriger**

---

### 6. **select_best_results.py - Utilitaire Isolé (161 lignes)**

**Rôle**: Génère `best_results_summary.json` en analysant tous les JSONs de performance.

**Problème**: 
- Pas appelé par les runners!
- Doit être lancé manuellement avant les visualizations
- **Devrait faire partie du pipeline automatique**

**Recommandation**: **Intégrer dans le runner principal comme étape 0**

---

### 7. **Incohérence dans --best-only Flag**

#### `run_all_visualizations.py`:
```python
parser.add_argument('--best-only', action='store_true', default=True,
                    help='Run reduced visualizations using only best thresholds (default: True)')

if args.best_only:
    import performance_comparison_best as pc_best
    pc_best.main()
else:
    performance_comparison.main()  # ← Lance l'ancien script!
```

**Problème**: Flag `--best-only` est True par défaut, donc:
- `python run_all_visualizations.py` → utilise `performance_comparison_best.py` ✅
- `python run_all_visualizations.py --no-best-only` → utilise l'ancien `performance_comparison.py` ⚠️

**Confusion**: L'ancien script est accessible mais découragé

---

## ✅ CE QUI FONCTIONNE BIEN

### Scripts "Modern" (bien conçus):
1. **performance_comparison_best.py** (359 lignes)
   - Charge `best_results_summary.json`
   - Génère plots clairs et utiles
   - Utilise `config.PERFORMANCE_DIR` centralisé
   - Approche "best threshold only" évite explosion de PNGs

2. **modern_dataset_analysis.py** (300 lignes)
   - Analyse composition dataset
   - Plots clairs (distributions, SNR, catégories)
   - Auto-suffisant

3. **modern_audio_examples.py** (298 lignes)
   - Génère HTML + audio embeddings
   - Copie fichiers WAV
   - **Bonne approche**: fichiers séparés, pas de base64
   - Structure propre: outputs/audio_examples/

4. **modern_threshold_calibration.py** (302 lignes)
   - Recommandations threshold intelligentes
   - Multi-critères (F1, accuracy, balanced)
   - Génère JSON + TXT + PNG

5. **model_comparison_plots.py** (259 lignes)
   - Comparaisons visuelles entre modèles
   - Bien structuré, réutilisable

6. **snr_distribution.py** (218 lignes)
   - Analyse SNR par catégorie/distance
   - Visualisation claire

7. **threshold_calibration_comparison.py** (180 lignes)
   - Plots impact des thresholds
   - Complémentaire à modern_threshold_calibration

---

## 🎯 RECOMMENDATIONS PRIORITAIRES

### Priority 1: **Simplifier les Runners (HIGH)**
**Action**: Fusionner les deux runners en UN SEUL moderne

**Nouveau fichier**: `run_visualizations.py` (remplace les 2 actuels)

**Approche**:
```python
#!/usr/bin/env python3
"""
Unified Visualization Runner
Lance toutes les visualisations modernes dans le bon ordre.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Import des scripts modernes uniquement
import select_best_results
import performance_comparison_best
import threshold_calibration_comparison
import model_comparison_plots
import snr_distribution
import modern_dataset_analysis
import modern_threshold_calibration
import generate_html_report

def main():
    """Pipeline de visualisation complet."""
    
    print("="*80)
    print("  VISUALIZATION PIPELINE")
    print("="*80)
    
    steps = [
        ("Select Best Results", select_best_results.main),
        ("Performance Comparison", performance_comparison_best.main),
        ("Threshold Calibration Comparison", threshold_calibration_comparison.main),
        ("Model Comparison Plots", model_comparison_plots.main),
        ("SNR Distribution", snr_distribution.main),
        ("Dataset Analysis", modern_dataset_analysis.main),
        ("Threshold Calibration", modern_threshold_calibration.main),
        ("HTML Report", generate_html_report.main),
    ]
    
    for i, (name, func) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {name}...")
        try:
            func()
            print(f"✓ {name} completed")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    print("\n" + "="*80)
    print("✓ Pipeline complete!")
```

**Bénéfices**:
- ✅ Un seul runner à maintenir
- ✅ Imports directs (plus rapide que subprocess)
- ✅ Pipeline clair et ordonné
- ✅ Inclut `select_best_results` automatiquement
- ✅ Numérotation cohérente

---

### Priority 2: **Déprécier performance_comparison.py (MEDIUM)**

**Actions**:
1. Renommer: `performance_comparison.py` → `_deprecated_performance_comparison.py`
2. Créer stub avec message de redirection:
```python
#!/usr/bin/env python3
"""
DEPRECATED: Use performance_comparison_best.py instead
This script is kept for backward compatibility only.
"""
import sys
print("WARNING: This script is deprecated!")
print("Use: python performance_comparison_best.py")
print("Or run the full pipeline: python run_visualizations.py")
sys.exit(1)
```
3. Mettre à jour `quick_viz.py` pour pointer vers `performance_comparison_best.py`

---

### Priority 3: **Refactorer generate_html_report.py (MEDIUM)**

**Actions**:
1. **Remplacer base64 par chemins relatifs**:
```python
# Au lieu de:
encoded_images['threshold'] = encode_image_to_base64(image_path)
html += f'<img src="data:image/png;base64,{encoded_images['threshold']}">'

# Faire:
shutil.copy(image_path, output_dir / "images" / "threshold_calibration.png")
html += f'<img src="./images/threshold_calibration.png">'
```

2. **Extraire le CSS dans fichier séparé**:
```
outputs/
  report.html
  style.css
  images/
    threshold_calibration.png
    model_comparison.png
    ...
```

3. **Utiliser template engine** (Jinja2 ou simple format):
```python
from string import Template

template = Template(Path("report_template.html").read_text())
html = template.substitute(
    title="UAV Performance Report",
    date=datetime.now().strftime("%Y-%m-%d"),
    # ...
)
```

**Bénéfices**:
- ✅ HTML ~10x plus petit
- ✅ Images réutilisables séparément
- ✅ CSS modifiable sans toucher au code Python
- ✅ Plus rapide à charger
- ✅ Meilleure séparation des responsabilités

---

### Priority 4: **Mettre à jour quick_viz.py (LOW)**

**Action**: Pointer vers les scripts modernes
```python
# Au lieu de:
SCRIPT_PATH = Path(__file__).parent / "performance_comparison.py"

# Utiliser:
SCRIPT_PATH = Path(__file__).parent / "performance_comparison_best.py"

# Ou mieux: lancer le runner complet
SCRIPT_PATH = Path(__file__).parent / "run_visualizations.py"
```

---

### Priority 5: **Corriger Numérotation Pipeline (LOW)**

Dans `run_enhanced_visualizations.py` (ou le nouveau runner):
```python
# Retirer les numéros ou corriger:
pipeline = [
    ("Select Best Results", select_best_results.main),  # Nouveau step 0
    ("Performance Analysis", performance_comparison_best.main),
    ("Threshold Calibration Comparison", threshold_calibration_comparison.main),
    ("Model Comparison", model_comparison_plots.main),
    ("SNR Distribution", snr_distribution.main),
    ("Dataset Analysis", modern_dataset_analysis.main),
    ("Modern Threshold Calibration", modern_threshold_calibration.main),
    ("HTML Report", generate_html_report.main),
]
```

---

## 📋 PLAN D'ACTION DÉTAILLÉ

### Phase 1: Cleanup (1-2h)
- [ ] Créer `run_visualizations.py` (nouveau runner unifié)
- [ ] Renommer `performance_comparison.py` → `_deprecated_performance_comparison.py`
- [ ] Créer stub de redirection dans ancien fichier
- [ ] Mettre à jour `quick_viz.py` pour pointer vers nouveau runner
- [ ] Tester le nouveau pipeline complet

### Phase 2: Amélioration HTML (2-3h)
- [ ] Extraire CSS dans `style.css` séparé
- [ ] Créer dossier `outputs/images/` pour les images
- [ ] Modifier `generate_html_report.py` pour copier images au lieu de base64
- [ ] Créer template HTML séparé (optionnel, ou garder string formatté simple)
- [ ] Tester génération HTML et vérifier taille fichier

### Phase 3: Documentation (30min)
- [ ] Mettre à jour README.md pour refléter nouveau workflow
- [ ] Ajouter exemples d'usage du nouveau runner
- [ ] Documenter les scripts obsolètes
- [ ] Créer guide de migration

### Phase 4: Validation (1h)
- [ ] Lancer pipeline complet: `python run_visualizations.py`
- [ ] Vérifier tous les outputs générés
- [ ] Comparer avec anciens outputs (qualité identique?)
- [ ] Vérifier taille HTML report (devrait être ~10x plus petit)
- [ ] Tester quick_viz presets

---

## 📊 MÉTRIQUES AVANT/APRÈS

### Avant Refactoring:
- **2 runners** avec approches différentes (imports vs subprocess)
- **2 scripts performance** (640 + 359 lignes) redondants
- **HTML report**: ~2-5 MB (avec base64)
- **Pipeline**: 7-8 étapes (numérotation incohérente)
- **Scripts obsolètes**: 1 gros (performance_comparison.py)
- **Confusion**: Quel runner utiliser? Quel script performance?

### Après Refactoring:
- **1 runner** unifié, approche consistente
- **1 script performance** (performance_comparison_best.py)
- **HTML report**: ~200-500 KB (chemins relatifs)
- **Pipeline**: 8 étapes numérotées (inclut select_best_results)
- **Scripts obsolètes**: Clairement marqués (_deprecated)
- **Clarté**: Un seul point d'entrée, workflow évident

---

## 🔍 SCRIPTS À GARDER (Aucun changement)

Ces scripts sont bien conçus et ne nécessitent pas de modifications:
- ✅ `performance_comparison_best.py`
- ✅ `modern_dataset_analysis.py`
- ✅ `modern_audio_examples.py`
- ✅ `modern_threshold_calibration.py`
- ✅ `model_comparison_plots.py`
- ✅ `snr_distribution.py`
- ✅ `threshold_calibration_comparison.py`
- ✅ `select_best_results.py` (juste l'intégrer au pipeline)

---

## 📝 NOTES ADDITIONNELLES

### Architecture Actuelle (Confuse):
```
run_all_visualizations.py ───► performance_comparison.py (640 lignes, legacy)
                          └───► modern_* scripts

run_enhanced_visualizations.py ───► subprocess tous les scripts
                                     (inclut performance_comparison_best.py)

quick_viz.py ───► performance_comparison.py (legacy)
```

### Architecture Proposée (Claire):
```
run_visualizations.py ───► select_best_results (nouveau step 0)
                      ├───► performance_comparison_best
                      ├───► threshold_calibration_comparison
                      ├───► model_comparison_plots
                      ├───► snr_distribution
                      ├───► modern_dataset_analysis
                      ├───► modern_threshold_calibration
                      └───► generate_html_report (refactoré)

quick_viz.py ───► run_visualizations.py (avec presets)

_deprecated_performance_comparison.py ───► (stub avec message)
```

---

## ⚡ QUICK WINS (Faciles et rapides)

1. **Renommer runners** (2 min):
   - `run_all_visualizations.py` → `_old_run_all.py`
   - `run_enhanced_visualizations.py` → `_old_run_enhanced.py`

2. **Créer nouveau runner** (15 min):
   - Copier structure de `run_all_visualizations.py`
   - Remplacer `performance_comparison` par `performance_comparison_best`
   - Ajouter `select_best_results` en step 0
   - Retirer numéros des steps

3. **Stub deprecation** (5 min):
   - Créer `_deprecated_performance_comparison.py`
   - Message de redirection

4. **Mettre à jour README** (10 min):
   - Pointer vers nouveau runner
   - Marquer anciens scripts comme obsolètes

**Total: ~30 min pour quick wins majeurs!**

---

**Fin de l'audit**
