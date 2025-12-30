#!/bin/bash

# VFX Motion Capture Setup Script
# This script sets up the development environment

set -e

echo "==================================="
echo "VFX Motion Capture Setup"
echo "==================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Please do not run this script as root${NC}"
    exit 1
fi

# Check for required tools
check_tool() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 found"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        return 1
    fi
}

echo "Checking required tools..."
MISSING_TOOLS=0

check_tool python3 || MISSING_TOOLS=1
check_tool pip || MISSING_TOOLS=1
check_tool node || MISSING_TOOLS=1
check_tool npm || MISSING_TOOLS=1
check_tool docker || MISSING_TOOLS=1
check_tool ffmpeg || MISSING_TOOLS=1

if [ $MISSING_TOOLS -eq 1 ]; then
    echo ""
    echo -e "${YELLOW}Some tools are missing. Please install them before continuing.${NC}"
    echo ""
    echo "Installation guides:"
    echo "  - Python: https://www.python.org/downloads/"
    echo "  - Node.js: https://nodejs.org/"
    echo "  - Docker: https://docs.docker.com/get-docker/"
    echo "  - FFmpeg: https://ffmpeg.org/download.html"
    exit 1
fi

echo ""
echo "All required tools found!"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Check Node version
NODE_VERSION=$(node --version)
echo "Node version: $NODE_VERSION"

# Check for NVIDIA GPU
echo ""
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")
    if [ -n "$GPU_INFO" ]; then
        echo -e "${GREEN}✓${NC} NVIDIA GPU found: $GPU_INFO"
    fi
else
    echo -e "${YELLOW}⚠${NC} nvidia-smi not found. GPU acceleration may not be available."
fi

# Create directories
echo ""
echo "Creating directories..."
mkdir -p uploads outputs temp models/diffusion_models models/text_encoders models/vae models/liveportrait models/insightface

# Setup environment file
echo ""
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo -e "${GREEN}✓${NC} .env file created"
else
    echo -e "${YELLOW}⚠${NC} .env file already exists, skipping"
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
cd backend
pip install -r requirements.txt
cd ..

# Install Node.js dependencies
echo ""
echo "Installing Node.js dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "==================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "==================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Edit .env file with your settings"
echo ""
echo "2. Download AI models:"
echo "   - Wan 2.1 VACE: huggingface-cli download Wan-AI/Wan2.1-VACE-14B"
echo "   - Wan 2.6 R2V: huggingface-cli download Wan-AI/Wan2.6-R2V-14B"
echo "   - LivePortrait: git clone https://github.com/KwaiVGI/LivePortrait"
echo ""
echo "3. Start ComfyUI:"
echo "   cd comfyui && docker-compose up"
echo ""
echo "4. Start development servers:"
echo "   make dev"
echo ""
echo "5. Open http://localhost:3000 in your browser"
echo ""
