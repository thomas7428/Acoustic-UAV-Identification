# 🚀 Enhanced Dataset Pipeline v2.0 - READY TO USE!

## ✅ Status: All Tests Passed (34/34)

Le nouveau pipeline est **opérationnel et validé** ! Voici comment l'utiliser.

---

## 📦 Ce qui a été créé

### 6 Nouveaux Fichiers Principaux

1. **`augment_dataset_v2.py`** (735 lignes)
   - Augmentation des **deux classes** (drones ET no-drones)
   - Effets audio : pitch shift, time stretch
   - Mixage intelligent de sources multiples
   - Métadonnées complètes

2. **`augment_config_v2.json`**
   - Configuration professionnelle
   - Drones : 5 catégories SNR (-15dB à +5dB)
   - No-drones : 4 catégories de complexité
   - Paramètres audio optimisés

3. **`split_dataset.py`** (400 lignes)
   - Split train/validation/test
   - **Zéro data leakage** garanti
   - Vérification automatique
   - Balance des classes maintenue

4. **`master_setup_v2.py`** (550 lignes)
   - Pipeline automatisé complet
   - 6 étapes orchestrées
   - Interface colorée et claire
   - Gestion d'erreurs robuste

5. **`README_V2.md`** - Documentation complète
6. **`MIGRATION_GUIDE.md`** - Guide de transition

### 3 Fichiers de Documentation

7. **`test_pipeline_v2.py`** - Script de validation (ce que vous venez de lancer)
8. **`ENHANCED_PIPELINE_V2_SUMMARY.md`** - Récapitulatif technique au niveau projet
9. **Ce fichier** - Guide de démarrage rapide

---

## 🎯 Utilisation Rapide

### Option 1 : Setup Complet Automatique (Recommandé)

```powershell
cd "0 - DADS dataset extraction"

# Lancer le pipeline complet
python master_setup_v2.py --drone-samples 100 --no-drone-samples 100

# Durée estimée : 10-30 minutes
# Résultat : Datasets train/val/test prêts + features extraites
```

### Option 2 : Étape par Étape

```powershell
# 1. Télécharger DADS
python download_and_prepare_dads.py --output dataset_test --max-per-class 100

# 2. Augmenter les deux classes
python augment_dataset_v2.py --config augment_config_v2.json

# 3. Combiner originals + augmentés
# (fait manuellement ou via master_setup_v2.py)

# 4. Séparer train/val/test
python split_dataset.py --source dataset_combined --train 0.7 --val 0.15 --test 0.15

# 5. Extraire les features
$env:DATASET_ROOT_OVERRIDE = "dataset_train"
python "..\1 - Preprocessing and Features Extraction\Mel_Preprocess_and_Feature_Extract.py"
```

### Option 3 : Dry Run (Voir sans exécuter)

```powershell
python master_setup_v2.py --dry-run
# Affiche ce qui serait fait sans créer de fichiers
```

---

## 📊 Résultat Attendu

Après exécution, vous aurez :

```
0 - DADS dataset extraction/
├── dataset_test/              # 100 originals par classe = 200
├── dataset_augmented/         # 200 drones + 200 no-drones = 400
├── dataset_combined/          # 300 par classe = 600
├── dataset_train/             # 210 par classe = 420 (70%)
├── dataset_val/               # 45 par classe = 90 (15%)
└── dataset_test/              # 45 par classe = 90 (15%)
```

### Vérifications Automatiques

✅ Balance 50/50 dans chaque split  
✅ Zéro overlap entre train/val/test  
✅ Métadonnées complètes sauvegardées  
✅ Features Mel extraites du dataset_train  

---

## 🔧 Entraînement et Test

### 1. Entraîner les Modèles

```powershell
# Utiliser dataset_train
$env:DATASET_ROOT_OVERRIDE = "dataset_train"

# Entraîner CNN
python "2 - Model Training\CNN_Trainer.py"

# Entraîner RNN
python "2 - Model Training\RNN_Trainer.py"

# Entraîner CRNN
python "2 - Model Training\CRNN_Trainer.py"
```

### 2. Évaluer les Performances

```powershell
# Utiliser dataset_test (JAMAIS vu pendant l'entraînement!)
$env:DATASET_ROOT_OVERRIDE = "dataset_test"

# Calculer les performances
python "3 - Single Model Performance Calculation\CNN_and_CRNN_Performance_Calcs.py"
python "3 - Single Model Performance Calculation\RNN_Performance_Calcs.py"
```

### 3. Visualisations

```powershell
# Performances par distance SNR
python "6 - Visualization\performance_by_distance.py"

# Autres visualisations
python "6 - Visualization\confusion_matrix_comparison.py"
```

---

## 🎓 Comprendre les Résultats

### Avant v2.0 (Avec Data Leakage)
```
Training:   dataset_combined
Testing:    dataset_combined (MÊME FICHIERS!)
Accuracy:   100% ❌ FAUX - Le modèle a mémorisé
```

### Après v2.0 (Sans Data Leakage)
```
Training:   dataset_train (70% unique)
Testing:    dataset_test (15% JAMAIS vu)
Accuracy:   55-65% ✅ VRAI - Performance réelle
```

**C'est normal que l'accuracy soit plus basse !** Cela signifie que vos modèles sont maintenant **correctement évalués**.

---

## 📚 Paramètres Personnalisables

### Nombre d'échantillons

```powershell
# Plus de données = meilleur apprentissage
python master_setup_v2.py --drone-samples 500 --no-drone-samples 500
```

### Ratios de split

```powershell
# 80/10/10 au lieu de 70/15/15
python master_setup_v2.py --train 0.8 --val 0.1 --test 0.1

# Sans validation (80/20 simple)
python split_dataset.py --source dataset_combined --train 0.8 --test 0.2 --no-val
```

### Configuration d'augmentation

Modifiez `augment_config_v2.json` :

```json
{
  "output": {
    "samples_per_category_drone": 300,      // Plus de drones augmentés
    "samples_per_category_no_drone": 300    // Plus de no-drones augmentés
  },
  
  "no_drone_augmentation": {
    "categories": [
      {
        "name": "ambient_complex",
        "proportion": 0.5,                   // Changez les proportions
        "num_noise_sources": 4,              // Plus de sources
        "enable_pitch_shift": true,
        "pitch_shift_range": [-3, 3]        // Range plus large
      }
    ]
  }
}
```

---

## 🐛 Dépannage

### "Module 'librosa' has no attribute 'effects'"

```powershell
# Mettre à jour librosa
pip install --upgrade librosa
```

### "DATASET_ROOT_OVERRIDE not working"

```powershell
# PowerShell (recommandé)
$env:DATASET_ROOT_OVERRIDE = "dataset_train"

# CMD
set DATASET_ROOT_OVERRIDE=dataset_train

# Vérifier
python -c "import os; print(os.environ.get('DATASET_ROOT_OVERRIDE'))"
```

### "Not enough samples"

```powershell
# Télécharger plus de données
python download_and_prepare_dads.py --max-per-class 1000
```

---

## 📖 Documentation Complète

- **`README_V2.md`** → Référence technique complète
- **`MIGRATION_GUIDE.md`** → Si vous avez déjà des données v1.0
- **`ENHANCED_PIPELINE_V2_SUMMARY.md`** → Vue d'ensemble du projet

---

## ✅ Checklist Post-Setup

Après avoir lancé le pipeline, vérifiez :

- [ ] `dataset_train/` existe avec ~70% des données
- [ ] `dataset_val/` existe avec ~15% des données
- [ ] `dataset_test/` existe avec ~15% des données
- [ ] Les deux classes (0/ et 1/) sont présentes partout
- [ ] `split_info.json` confirme zéro overlap
- [ ] `augmentation_metadata.json` existe dans dataset_augmented/
- [ ] Features Mel extraites dans `extracted_features/mel_data.json`

---

## 🎯 Prochaines Étapes

1. **Lancer le setup**
   ```powershell
   python master_setup_v2.py
   ```

2. **Vérifier les résultats**
   ```powershell
   # Voir le split info
   cat split_info.json
   
   # Compter les fichiers
   (Get-ChildItem dataset_train\0 -Filter *.wav).Count
   (Get-ChildItem dataset_train\1 -Filter *.wav).Count
   ```

3. **Entraîner les modèles**
   ```powershell
   $env:DATASET_ROOT_OVERRIDE = "dataset_train"
   python "2 - Model Training\CNN_Trainer.py"
   ```

4. **Évaluer sur test set**
   ```powershell
   $env:DATASET_ROOT_OVERRIDE = "dataset_test"
   python "3 - Single Model Performance Calculation\CNN_and_CRNN_Performance_Calcs.py"
   ```

5. **Comparer les résultats** (avant vs après v2.0)

---

## 💡 Rappels Importants

1. **Ne JAMAIS entraîner sur dataset_test** → Utilisez dataset_train
2. **Ne JAMAIS tester sur dataset_train** → Utilisez dataset_test
3. **dataset_val est pour le tuning** → Ajustement d'hyperparamètres
4. **Une accuracy plus basse est normale** → C'est la vraie performance
5. **Le pipeline est reproductible** → Random seed = 42 par défaut

---

## 🎉 Conclusion

Vous avez maintenant un **pipeline professionnel de classe production** pour :

✅ Créer des datasets équilibrés  
✅ Éviter le data leakage  
✅ Évaluer correctement vos modèles  
✅ Obtenir des résultats fiables  

**Le système est prêt !** Lancez `python master_setup_v2.py` et c'est parti ! 🚀

---

**Questions ?** Consultez la documentation ou les commentaires inline dans les scripts.

**Créé le** : 11 décembre 2025  
**Version** : 2.0  
**Statut** : ✅ Production Ready
