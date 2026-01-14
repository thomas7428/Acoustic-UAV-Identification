#!/usr/bin/env python3
"""Test simple pour vérifier que TensorFlow détecte le GPU AMD via ROCm"""

import tensorflow as tf
import sys

print("=" * 60)
print("TEST GPU AMD RX7600 avec TensorFlow-ROCm")
print("=" * 60)

print(f"\nTensorFlow version: {tf.__version__}")
print(f"Built with ROCm: {tf.test.is_built_with_rocm()}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

print("\n" + "=" * 60)
print("DEVICES DETECTES:")
print("=" * 60)

# Lister tous les devices
all_devices = tf.config.list_physical_devices()
print(f"\nTotal devices: {len(all_devices)}")
for device in all_devices:
    print(f"  - {device}")

# Vérifier spécifiquement les GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"\n{'=' * 60}")
print(f"GPUs détectés: {len(gpus)}")
print("=" * 60)

if gpus:
    print("✅ GPU(s) détecté(s) !")
    for i, gpu in enumerate(gpus):
        print(f"\nGPU {i}: {gpu}")
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"  Détails: {details}")
        except Exception as e:
            print(f"  (Impossible d'obtenir les détails: {e})")
    
    # Test rapide de calcul sur GPU
    print(f"\n{'=' * 60}")
    print("TEST DE CALCUL SUR GPU")
    print("=" * 60)
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"✅ Multiplication matricielle réussie sur GPU!")
            print(f"Résultat:\n{c.numpy()}")
    except Exception as e:
        print(f"❌ Erreur lors du calcul sur GPU: {e}")
else:
    print("❌ AUCUN GPU DÉTECTÉ")
    print("\nTensorFlow utilise seulement le CPU.")
    sys.exit(1)

print(f"\n{'=' * 60}")
print("✅ TEST TERMINÉ AVEC SUCCÈS!")
print("=" * 60)
