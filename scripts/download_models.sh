#!/bin/bash

# Model Download Script for VFX Motion Capture
# Downloads required AI models from HuggingFace

set -e

echo "==================================="
echo "VFX Motion Capture Model Downloader"
echo "==================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Base directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${YELLOW}Installing huggingface_hub...${NC}"
    pip install huggingface_hub[cli]
fi

# Create directories
mkdir -p "$MODELS_DIR/diffusion_models"
mkdir -p "$MODELS_DIR/text_encoders"
mkdir -p "$MODELS_DIR/vae"
mkdir -p "$MODELS_DIR/liveportrait"
mkdir -p "$MODELS_DIR/insightface"

# Function to download model
download_model() {
    local repo=$1
    local local_dir=$2
    local description=$3

    echo ""
    echo -e "${BLUE}Downloading $description...${NC}"
    echo "Repository: $repo"
    echo "Destination: $local_dir"
    echo ""

    if [ -d "$local_dir" ] && [ "$(ls -A $local_dir 2>/dev/null)" ]; then
        echo -e "${YELLOW}Directory not empty. Skip? (y/n)${NC}"
        read -r skip
        if [ "$skip" = "y" ]; then
            echo "Skipping..."
            return
        fi
    fi

    huggingface-cli download "$repo" --local-dir "$local_dir" --local-dir-use-symlinks False
    echo -e "${GREEN}✓ Downloaded $description${NC}"
}

# Menu
echo "Select models to download:"
echo ""
echo "1) Wan 2.1 VACE 14B (Pose/Motion Transfer) - ~28GB"
echo "2) Wan 2.6 R2V 14B (Reference-to-Video) - ~28GB"
echo "3) Text Encoder (UMT5-XXL) - ~20GB"
echo "4) VAE (Wan 2.1) - ~400MB"
echo "5) InsightFace (Face Detection) - ~1GB"
echo "6) All of the above"
echo "7) Exit"
echo ""
echo -n "Enter choice [1-7]: "
read -r choice

case $choice in
    1)
        download_model "Wan-AI/Wan2.1-VACE-14B" "$MODELS_DIR/diffusion_models/wan2.1_vace" "Wan 2.1 VACE"
        ;;
    2)
        download_model "Wan-AI/Wan2.6-R2V-14B" "$MODELS_DIR/diffusion_models/wan2.6_r2v" "Wan 2.6 R2V"
        ;;
    3)
        download_model "google/umt5-xxl" "$MODELS_DIR/text_encoders/umt5_xxl" "UMT5-XXL Text Encoder"
        ;;
    4)
        echo -e "${BLUE}Downloading Wan VAE...${NC}"
        cd "$MODELS_DIR/vae"
        wget -nc "https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1_VAE.safetensors" || true
        echo -e "${GREEN}✓ Downloaded Wan VAE${NC}"
        ;;
    5)
        echo -e "${BLUE}Setting up InsightFace...${NC}"
        pip install insightface onnxruntime-gpu
        echo "InsightFace models will be downloaded on first use"
        echo -e "${GREEN}✓ InsightFace installed${NC}"
        ;;
    6)
        download_model "Wan-AI/Wan2.1-VACE-14B" "$MODELS_DIR/diffusion_models/wan2.1_vace" "Wan 2.1 VACE"
        download_model "Wan-AI/Wan2.6-R2V-14B" "$MODELS_DIR/diffusion_models/wan2.6_r2v" "Wan 2.6 R2V"
        download_model "google/umt5-xxl" "$MODELS_DIR/text_encoders/umt5_xxl" "UMT5-XXL Text Encoder"

        echo -e "${BLUE}Downloading Wan VAE...${NC}"
        cd "$MODELS_DIR/vae"
        wget -nc "https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1_VAE.safetensors" || true

        pip install insightface onnxruntime-gpu
        echo -e "${GREEN}✓ All models downloaded!${NC}"
        ;;
    7)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo "==================================="
echo -e "${GREEN}Download Complete!${NC}"
echo "==================================="
echo ""
echo "Models are stored in: $MODELS_DIR"
echo ""
echo "For LivePortrait, please clone manually:"
echo "  git clone https://github.com/KwaiVGI/LivePortrait $MODELS_DIR/liveportrait"
echo ""
