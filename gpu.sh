#!/bin/bash
# Script unifié pour utiliser le GPU AMD RX7600 avec TensorFlow-ROCm
# Usage: 
#   ./gpu.sh --test                    # Tester le GPU
#   ./gpu.sh --script mon_script.py    # Exécuter un script avec GPU

set -e

IMAGE="acoustic-uav-rocm"
FALLBACK_IMAGE="docker.io/rocm/tensorflow:latest"
GFX_VERSION="11.0.2"  # Pour RX7600 (gfx1102)

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo -e "${BLUE}Usage:${NC}"
    echo "  $0 --test                     Test du GPU"
    echo "  $0 --script <fichier.py>      Exécuter un script Python avec GPU"
    echo ""
    echo -e "${BLUE}Exemples:${NC}"
    echo "  $0 --test"
    echo "  $0 --script '2 - Model Training/EfficientNet_Trainer.py'"
    exit 1
}

check_image() {
    echo -e "${BLUE}🔍 Vérification de l'image TensorFlow-ROCm...${NC}"
    if podman images | grep -q "acoustic-uav-rocm"; then
        echo -e "${GREEN}✅ Image personnalisée présente (acoustic-uav-rocm)${NC}"
        IMAGE="acoustic-uav-rocm"
        return 0
    elif podman images | grep -q "rocm/tensorflow"; then
        echo -e "${YELLOW}⚠️  Image de base présente, mais pas la version personnalisée${NC}"
        echo -e "${BLUE}Reconstruction de l'image personnalisée...${NC}"
        cd "$(dirname "$0")"
        podman build -t acoustic-uav-rocm -f Dockerfile.rocm .
        IMAGE="acoustic-uav-rocm"
        return 0
    else
        echo -e "${YELLOW}⚠️  Image TensorFlow-ROCm non trouvée${NC}"
        echo ""
        echo -e "${BLUE}L'image fait ~28 GB et va prendre plusieurs minutes à télécharger.${NC}"
        read -p "Voulez-vous l'installer maintenant? (o/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Oo]$ ]]; then
            echo -e "${BLUE}📦 Téléchargement et construction de l'image...${NC}"
            cd "$(dirname "$0")"
            podman pull "$FALLBACK_IMAGE"
            podman build -t acoustic-uav-rocm -f Dockerfile.rocm .
            IMAGE="acoustic-uav-rocm"
            echo -e "${GREEN}✅ Image installée avec succès${NC}"
        else
            echo -e "${RED}❌ Installation annulée${NC}"
            exit 1
        fi
    fi
}

check_gpu() {
    echo -e "${BLUE}🎮 Vérification du GPU AMD...${NC}"
    if ! command -v rocm-smi &> /dev/null; then
        echo -e "${RED}❌ rocm-smi non trouvé. ROCm est-il installé?${NC}"
        exit 1
    fi
    
    rocm-smi --showproductname 2>/dev/null || {
        echo -e "${RED}❌ Impossible de détecter le GPU${NC}"
        exit 1
    }
    echo -e "${GREEN}✅ GPU détecté${NC}"
    echo ""
}

test_gpu() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}    TEST GPU AMD RX7600 + TensorFlow${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
    
    check_image
    check_gpu
    
    echo -e "${BLUE}🧪 Test 1: Détection GPU (rapide)...${NC}"
    
    timeout 45 podman run --rm \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --security-opt seccomp=unconfined \
        -e HSA_OVERRIDE_GFX_VERSION="$GFX_VERSION" \
        -e TF_CPP_MIN_LOG_LEVEL=0 \
        "$IMAGE" \
        python3 -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print(f'\\n✅ {len(gpus)} GPU(s) détecté(s)'); [print(f'  - {g}') for g in gpus]; exit(0 if gpus else 1)" \
    && GPU_TEST_RESULT=$? || GPU_TEST_RESULT=$?
    
    if [ $GPU_TEST_RESULT -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Test GPU réussi!${NC}"
        echo ""
        echo -e "${YELLOW}ℹ️  HSA_OVERRIDE_GFX_VERSION=$GFX_VERSION est nécessaire car:${NC}"
        echo -e "   Votre RX7600 est gfx1102 (RDNA 3) mais TensorFlow-ROCm"
        echo -e "   a été compilé pour des architectures antérieures."
        echo -e "   L'override force la compatibilité.${NC}"
    elif [ $GPU_TEST_RESULT -eq 124 ]; then
        echo -e "${RED}❌ Timeout - Le conteneur prend trop de temps à démarrer${NC}"
        echo -e "${YELLOW}   Cela peut arriver lors du premier lancement${NC}"
        exit 1
    else
        echo -e "${RED}❌ Test GPU échoué (code: $GPU_TEST_RESULT)${NC}"
        exit 1
    fi
}

run_script() {
    local script="$1"
    
    if [ ! -f "$script" ]; then
        echo -e "${RED}❌ Erreur: Le fichier '$script' n'existe pas${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}    EXÉCUTION AVEC GPU AMD RX7600${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}📄 Script: $script${NC}"
    echo ""
    
    check_image
    check_gpu
    
    echo -e "${BLUE}🚀 Lancement...${NC}"
    echo ""
    
    podman run --rm -it \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --security-opt seccomp=unconfined \
        --security-opt label=disable \
        -v "$(pwd):/workspace:rw" \
        -w "/workspace" \
        -e HSA_OVERRIDE_GFX_VERSION="$GFX_VERSION" \
        "$IMAGE" \
        python3 "$script"
    
    echo ""
    echo -e "${GREEN}✅ Exécution terminée!${NC}"
}

# Parse arguments
case "${1:-}" in
    --test)
        test_gpu
        ;;
    --script)
        if [ -z "${2:-}" ]; then
            echo -e "${RED}❌ Erreur: --script nécessite un fichier${NC}"
            usage
        fi
        run_script "$2"
        ;;
    *)
        usage
        ;;
esac
