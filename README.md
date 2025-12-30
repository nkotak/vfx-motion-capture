# VFX Motion Capture

Real-time VFX motion capture system using state-of-the-art AI video generation models.

## Features

- **Image-to-Video Mode**: Upload a reference image and input video, then generate a new video where your character performs the motions
- **Real-Time Camera Mode**: Use your webcam/phone camera to control a character in real-time
- **Natural Language Prompts**: Describe what you want in plain English
- **Multiple AI Models**: Wan 2.6 R2V, Wan VACE, LivePortrait, Deep-Live-Cam

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose
- NVIDIA GPU (8GB+ VRAM recommended)
- FFmpeg

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vfx-motion-capture.git
cd vfx-motion-capture

# Run setup script
./scripts/setup.sh

# Or manually:
make setup
```

### Download Models

```bash
./scripts/download_models.sh
```

Or download manually:
- [Wan 2.1 VACE](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B)
- [Wan 2.6 R2V](https://huggingface.co/Wan-AI/Wan2.6-R2V-14B)
- [LivePortrait](https://github.com/KwaiVGI/LivePortrait)

### Start Development

```bash
# Start all services
make dev

# Or start individually:
make comfyui    # Start ComfyUI (required)
make backend    # Start FastAPI backend
make frontend   # Start Next.js frontend
make worker     # Start Celery worker
```

Open http://localhost:3000 in your browser.

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (Next.js)                              │
│  - Image/Video upload         - Real-time camera capture                    │
│  - Prompt input               - WebSocket progress updates                  │
│  - Video playback             - Mode selection                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND (FastAPI)                               │
│  - REST API endpoints         - Job queue management                        │
│  - WebSocket handlers         - File management                             │
│  - Prompt parsing             - Model routing                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│   Celery Workers        │ │   Redis                 │ │   ComfyUI               │
│   (Async Processing)    │ │   (Job Queue)           │ │   (AI Inference)        │
└─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
```

## Generation Modes

| Mode | Description | Speed | Quality |
|------|-------------|-------|---------|
| **Pose Transfer** | Transfer poses from video to character | Medium | High |
| **Motion Transfer** | Apply motion sequences to character | Medium | High |
| **Wan R2V** | Generate new video with character | Slow | Highest |
| **LivePortrait** | Real-time face animation | Fast | Good |
| **Face Swap** | Real-time face replacement | Fast | Good |

## API Reference

### Upload Endpoints

```
POST /api/upload/image    # Upload reference image
POST /api/upload/video    # Upload input video
```

### Generation Endpoints

```
POST /api/generate        # Start video generation
POST /api/generate/preview # Quick preview generation
GET  /api/generate/modes  # List available modes
```

### Job Endpoints

```
GET    /api/jobs              # List all jobs
GET    /api/jobs/{id}         # Get job details
GET    /api/jobs/{id}/progress # Get job progress
POST   /api/jobs/{id}/cancel  # Cancel job
DELETE /api/jobs/{id}         # Delete job
```

### WebSocket Endpoints

```
WS /ws/jobs/{job_id}     # Real-time job progress
WS /ws/realtime/{session_id}  # Real-time camera processing
```

## Project Structure

```
vfx-motion-capture/
├── backend/
│   ├── api/              # FastAPI routes and WebSocket handlers
│   ├── core/             # Configuration, models, exceptions
│   ├── services/         # Business logic (ComfyUI, video processing, etc.)
│   ├── workers/          # Celery async tasks
│   └── comfyui_workflows/ # ComfyUI workflow JSON files
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── hooks/        # Custom React hooks
│   │   ├── pages/        # Next.js pages
│   │   └── services/     # API client
├── comfyui/              # ComfyUI Docker setup
├── models/               # AI model weights (git-ignored)
├── scripts/              # Setup and utility scripts
├── docker-compose.yml    # Docker orchestration
└── Makefile              # Common commands
```

## Hardware Requirements

| Mode | Minimum GPU | Recommended GPU | VRAM |
|------|-------------|-----------------|------|
| Real-time | RTX 3060 | RTX 4070+ | 8GB |
| Standard | RTX 3080 | RTX 4080+ | 12GB |
| High Quality | RTX 4090 | A100/H100 | 24GB+ |

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# ComfyUI connection
COMFYUI_HOST=localhost
COMFYUI_PORT=8188

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# GPU settings
DEVICE=cuda
ENABLE_FP16=true
```

## Troubleshooting

### ComfyUI not connecting
- Ensure ComfyUI is running: `make comfyui`
- Check logs: `docker-compose logs comfyui`
- Verify port 8188 is accessible

### Out of GPU memory
- Use Draft quality mode
- Reduce video resolution/duration
- Enable FP16: `ENABLE_FP16=true`

### Slow generation
- Check GPU utilization: `nvidia-smi`
- Ensure CUDA is properly installed
- Try enabling xformers: `ENABLE_XFORMERS=true`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Wan Video](https://github.com/Wan-Video/Wan2.1) - Video generation models
- [LivePortrait](https://github.com/KwaiVGI/LivePortrait) - Portrait animation
- [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) - Real-time face swap
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - AI workflow engine
