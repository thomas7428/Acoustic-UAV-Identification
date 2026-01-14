# Configuration GPU AMD RX7600 pour TensorFlow

## ✅ Configuration Réussie

Votre GPU AMD RX7600 est maintenant configuré et fonctionnel pour l'entraînement TensorFlow !

### État actuel
- **GPU**: AMD Radeon RX 7600 (gfx1102)
- **ROCm**: 6.17.7 (pré-installé sur Bazzite)
- **TensorFlow-ROCm**: 2.20.0-dev0+selfbuilt (dans container Podman)
- **Utilisation GPU**: 100% pendant l'entraînement ✅

## Architecture de la solution

### Pourquoi un container ?
TensorFlow-ROCm n'est plus disponible via `pip install`. La seule méthode supportée est d'utiliser les images Docker/Podman officielles.

### Structure
```
Host (Bazzite)
├── ROCm 6.17.7 (drivers système)
├── Podman 5.7.1
└── Container: acoustic-uav-rocm
    ├── Base: docker.io/rocm/tensorflow:latest
    ├── TensorFlow 2.20.0 + ROCm
    └── Dépendances Python (sklearn, librosa, pandas, etc.)
```

## Fichiers créés

### 1. `Dockerfile.rocm`
Image personnalisée qui étend l'image ROCm officielle avec nos dépendances:
- scikit-learn
- librosa
- pandas, numpy, matplotlib, seaborn
- audiomentations

### 2. `container_requirements.txt`
Liste des packages Python à installer dans le container.

### 3. `gpu.sh`
Script autonome pour tester et utiliser le GPU:
```bash
./gpu.sh --test                              # Tester le GPU
./gpu.sh --script "mon_script.py"            # Exécuter un script avec GPU
```

### 4. Modifications de `run_full_pipeline.sh`
Le pipeline détecte automatiquement le GPU et l'utilise si disponible:
- Fonction `check_gpu_support()`: Détecte ROCm + Podman + GPU
- Fonction `run_python()`: Route les scripts via Podman si GPU disponible
- Transparent: aucun changement nécessaire dans vos scripts Python

## Utilisation

### Entraînement avec GPU (automatique)
```bash
./run_full_pipeline.sh --models EFFICIENTNET
```

Le pipeline détecte automatiquement le GPU et affiche:
```
[✓] GPU AMD détecté - Entraînement avec accélération GPU activé
[INFO] Exécution avec GPU: 2 - Model Training/EfficientNet_Trainer.py
```

### Vérifier l'utilisation du GPU
```bash
rocm-smi --showuse
# GPU[0] : GPU use (%): 100
```

### Entraîner un modèle spécifique
```bash
# Un seul modèle
./run_full_pipeline.sh --skip-dataset --skip-features --models EFFICIENTNET

# Plusieurs modèles en séquentiel
./run_full_pipeline.sh --models CNN,RNN,CRNN

# Plusieurs modèles en parallèle
./run_full_pipeline.sh --models CNN,RNN,CRNN --parallel
```

## Configuration technique

### HSA_OVERRIDE_GFX_VERSION=11.0.2
Votre RX7600 utilise l'architecture gfx1102 (RDNA 3), plus récente que celle pour laquelle l'image ROCm a été compilée. Cette variable d'environnement force la compatibilité.

### Devices et permissions
```bash
--device=/dev/kfd          # AMD GPU compute device
--device=/dev/dri          # Direct Rendering Infrastructure
--group-add video          # Groupe video pour accès GPU
--security-opt label=disable  # Nécessaire sur Fedora/Bazzite
```

### Volume mount
```bash
-v $(pwd):/workspace:rw    # Monte le répertoire projet dans le container
-w /workspace              # Définit le répertoire de travail
```

## Performance

### Avant (CPU)
- Python 3.12.2 + TensorFlow 2.20.0 (CPU uniquement)
- Pas d'accélération matérielle

### Après (GPU)
- AMD Radeon RX 7600 avec 7404 MB VRAM
- TensorFlow-ROCm 2.20.0 avec support XLA
- Utilisation GPU: 100% pendant l'entraînement
- **Accélération significative** 🚀

## Logs d'entraînement

Les logs montrent clairement l'utilisation du GPU:
```
I0000 ... gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 
  with 7404 MB memory:  -> device: 0, name: AMD Radeon RX 7600, pci bus id: 0000:00:07.0

I0000 ... service.cc:171] StreamExecutor device (0): AMD Radeon RX 7600, AMDGPU ISA version: gfx1102

I0000 ... device_compiler.h:196] Compiled cluster using XLA!
```

## Dépannage

### RNN/LSTM sur ROCm: "MIOpen only supports packed input output"

**Problème**: Les modèles utilisant LSTM/GRU (RNN, CRNN, Attention_CRNN) génèrent une erreur:
```
ROCm MIOpen only supports packed input output.
```

**Solution**: Ajouté `use_cudnn=False` aux couches LSTM pour forcer l'implémentation Python au lieu de MIOpen. Cela fonctionne sur GPU mais est légèrement plus lent que l'optimisation cuDNN/MIOpen.

**Fichiers modifiés**:
- [RNN_Trainer.py](2%20-%20Model%20Training/RNN_Trainer.py)
- [CRNN_Trainer.py](2%20-%20Model%20Training/CRNN_Trainer.py)
- [Attention_CRNN_Trainer.py](2%20-%20Model%20Training/Attention_CRNN_Trainer.py)

**Modèles non affectés** (fonctionnent à pleine vitesse GPU):
- ✅ CNN
- ✅ EfficientNet
- ✅ MobileNet
- ✅ Conformer
- ✅ TCN

### Le GPU n'est pas détecté
```bash
# Vérifier ROCm
rocm-smi

# Vérifier l'image Podman
podman images | grep acoustic-uav-rocm

# Reconstruire l'image
podman build -t acoustic-uav-rocm -f Dockerfile.rocm .
```

### Erreur "No module named 'sklearn'"
L'image de base ne contient pas nos dépendances. Il faut utiliser `acoustic-uav-rocm` et pas `docker.io/rocm/tensorflow:latest`.

```bash
# Vérifier que l'image personnalisée existe
podman images | grep acoustic-uav-rocm

# Si nécessaire, la reconstruire
podman build -t acoustic-uav-rocm -f Dockerfile.rocm .
```

### Container lent au premier lancement
XLA (Accelerated Linear Algebra) compile les kernels GPU optimisés au premier lancement. C'est normal et ne se produit qu'une fois. Les lancements suivants seront rapides.

## Commandes utiles

```bash
# Surveiller le GPU en temps réel
watch -n 1 rocm-smi --showuse

# Voir tous les GPU
rocm-smi --showproductname

# Température GPU
rocm-smi --showtemp

# Logs du pipeline
tail -f logs/pipeline_*.log

# Arrêter un entraînement
pkill -f "python3 /workspace"
# ou
kill <PID>

# Nettoyer les containers stoppés
podman container prune

# Nettoyer les images inutilisées
podman image prune
```

## Maintenance

### Mettre à jour l'image ROCm
```bash
podman pull docker.io/rocm/tensorflow:latest
podman build -t acoustic-uav-rocm -f Dockerfile.rocm .
```

### Ajouter des dépendances Python
1. Modifier `container_requirements.txt`
2. Reconstruire l'image:
   ```bash
   podman build -t acoustic-uav-rocm -f Dockerfile.rocm .
   ```

## Références

- [ROCm Documentation](https://rocmdocs.amd.com/)
- [TensorFlow ROCm Port](https://github.com/ROCm/tensorflow-upstream)
- [AMD GPU Support Matrix](https://github.com/ROCm/ROCm#hardware-and-software-support)

---

**Date**: 13 janvier 2026  
**Statut**: ✅ Opérationnel  
**GPU**: AMD Radeon RX 7600 (gfx1102)  
**System**: Bazzite Linux
