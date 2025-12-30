.PHONY: help install dev backend frontend worker docker-build docker-up docker-down clean test lint models

# Default target
help:
	@echo "VFX Motion Capture - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install all dependencies"
	@echo "  make install-backend Install Python dependencies"
	@echo "  make install-frontend Install Node.js dependencies"
	@echo "  make models         Download AI models"
	@echo ""
	@echo "Development:"
	@echo "  make dev            Start all services for development"
	@echo "  make backend        Start backend server"
	@echo "  make frontend       Start frontend server"
	@echo "  make worker         Start Celery worker"
	@echo "  make redis          Start Redis server"
	@echo "  make comfyui        Start ComfyUI"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start all Docker services"
	@echo "  make docker-down    Stop all Docker services"
	@echo "  make docker-logs    View Docker logs"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-backend   Run backend tests"
	@echo "  make lint           Run linters"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          Clean temporary files"
	@echo "  make clean-all      Clean everything including models"

# Installation
install: install-backend install-frontend
	@echo "Installation complete!"

install-backend:
	@echo "Installing Python dependencies..."
	cd backend && pip install -r requirements.txt

install-frontend:
	@echo "Installing Node.js dependencies..."
	cd frontend && npm install

# Development servers
dev:
	@echo "Starting development servers..."
	@make -j4 backend frontend worker redis

backend:
	@echo "Starting backend server..."
	cd backend && uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	@echo "Starting frontend server..."
	cd frontend && npm run dev

worker:
	@echo "Starting Celery worker..."
	cd backend && celery -A backend.workers.celery_app worker --loglevel=info

redis:
	@echo "Starting Redis..."
	docker run --rm -p 6379:6379 redis:7-alpine

comfyui:
	@echo "Starting ComfyUI..."
	cd comfyui && docker-compose up

# Docker commands
docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-up:
	@echo "Starting Docker services..."
	docker-compose up -d

docker-down:
	@echo "Stopping Docker services..."
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-restart:
	docker-compose restart

# Testing
test: test-backend

test-backend:
	@echo "Running backend tests..."
	cd backend && pytest -v

lint:
	@echo "Running linters..."
	cd backend && ruff check .
	cd frontend && npm run lint

# Model downloads
models: models-wan models-liveportrait models-insightface
	@echo "All models downloaded!"

models-wan:
	@echo "Downloading Wan models..."
	mkdir -p models/diffusion_models models/text_encoders models/vae
	@echo "Please download manually from HuggingFace:"
	@echo "  - Wan-AI/Wan2.1-VACE-14B"
	@echo "  - Wan-AI/Wan2.6-R2V-14B"

models-liveportrait:
	@echo "Downloading LivePortrait models..."
	mkdir -p models/liveportrait
	@echo "Please clone from: https://github.com/KwaiVGI/LivePortrait"

models-insightface:
	@echo "Downloading InsightFace models..."
	mkdir -p models/insightface
	@echo "Models will be downloaded automatically on first use"

# Cleaning
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".next" -exec rm -rf {} + 2>/dev/null || true
	rm -rf temp/* 2>/dev/null || true
	rm -rf .ruff_cache 2>/dev/null || true

clean-uploads:
	@echo "Cleaning uploads..."
	rm -rf uploads/* 2>/dev/null || true

clean-outputs:
	@echo "Cleaning outputs..."
	rm -rf outputs/* 2>/dev/null || true

clean-all: clean clean-uploads clean-outputs
	@echo "Cleaning models..."
	rm -rf models/* 2>/dev/null || true
	@echo "All clean!"

# Create necessary directories
setup-dirs:
	mkdir -p uploads outputs temp models/diffusion_models models/text_encoders models/vae models/liveportrait models/insightface

# Environment setup
setup-env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ".env file created. Please edit it with your settings."; \
	else \
		echo ".env file already exists."; \
	fi

# Full setup
setup: setup-dirs setup-env install
	@echo ""
	@echo "Setup complete! Next steps:"
	@echo "1. Edit .env file with your settings"
	@echo "2. Download AI models: make models"
	@echo "3. Start development: make dev"
