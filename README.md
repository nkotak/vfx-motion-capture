# VFX Motion Capture

Real-time and offline character motion transfer system built with a Next.js frontend and a FastAPI backend. The project supports both batch video generation workflows and a high-resolution realtime camera pipeline with workerized JPEG streaming, adaptive quality controls, and live telemetry.

## Highlights

- **Realtime camera mode** with binary JPEG WebSocket transport
- **High-resolution sessions** for 720p, 1080p, and 4K capture/output presets
- **Workerized realtime backend** with per-session worker affinity
- **Latest-frame-wins buffering** so stale frames can be dropped under load
- **Shared-memory transport** for large JPEG payloads between the WebSocket process and worker processes
- **Adaptive quality control** that can step down JPEG quality, tile mode, full-frame mode, and target FPS when latency exceeds budget
- **Per-session metrics and per-worker telemetry** for debugging realtime performance
- **Two realtime renderers**
  - `deep_live_cam`: realtime face swap
  - `liveportrait`: built-in landmark-driven portrait animation fallback, with optional external LivePortrait pipeline support if installed
- **Offline generation modes** for pose transfer, motion transfer, and Wan-based generation flows

## Realtime architecture at a glance

The current realtime stack is optimized around a server-style JPEG pipeline:

1. The browser captures camera frames and sends **binary JPEG blobs** over `WS /ws/realtime/{session_id}`
2. FastAPI receives frames and keeps a **latest-frame-wins** buffer per session
3. A dedicated realtime worker process decodes, processes, and re-encodes frames
4. Large payloads can move through **shared memory** instead of Python queue copies
5. The backend emits:
   - processed JPEG frames
   - adaptive update messages
   - metrics endpoints for session and worker telemetry

This repo now includes:

- worker pool startup in app lifespan
- shared-memory transport for large frames
- worker load / queue telemetry
- adaptive session quality policy
- tiled processing scaffolding
- optional frontend debug panel

## Supported modes

| Mode | Description | Realtime | Notes |
|------|-------------|----------|-------|
| `liveportrait` | Portrait animation driven by camera facial motion | Yes | Uses a built-in landmark-driven renderer; external LivePortrait integration is optional |
| `deep_live_cam` | Face swap using a cached source face and high-res aware detection | Yes | Preserves full-resolution output while using bounded analysis sizes |
| `vace_pose_transfer` | Offline pose transfer | No | Batch generation flow |
| `vace_motion_transfer` | Offline motion transfer | No | Batch generation flow |
| `wan_r2v` | Offline reference-to-video generation | No | Highest-cost batch mode |

## Hardware guidance

### Realtime

The realtime stack now supports both NVIDIA and Apple Silicon style deployments.

#### Apple Silicon
- Supported runtime: **MPS**
- Recommended local machine: **M2 Max / 96 GB RAM** or better for high-end local development
- Good starting target: **1080p / 24 FPS**
- 4K is best treated as a premium mode and should be validated against actual latency with the debug panel

#### NVIDIA
- Minimum practical realtime GPU: **RTX 3060 / 8 GB**
- Recommended for higher-end realtime: **RTX 4070+**
- High-end / multi-session experiments: **24 GB+ VRAM**

### Offline generation

| Tier | Suggested hardware |
|------|--------------------|
| Draft / preview | RTX 3060 or Apple Silicon with sufficient memory |
| Standard | RTX 3080 / 12 GB+ |
| High / heavy Wan workflows | RTX 4090 / A100 / H100 class |

## Quick start

### Prerequisites

- Python 3.10+
- Node.js 18+
- FFmpeg
- Docker and Docker Compose (optional but recommended for Redis and containerized services)

### Install

```bash
git clone https://github.com/yourusername/vfx-motion-capture.git
cd vfx-motion-capture

# Full setup
make setup

# Or step-by-step
make install-backend
make install-frontend
make setup-env
make setup-dirs
```

There is also a helper script:

```bash
./scripts/setup.sh
```

### Download models

```bash
make models
```

Or use:

```bash
./scripts/download_models.sh
```

Model sources referenced by this repo include:

- [Wan 2.1 VACE](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B)
- [Wan 2.6 R2V](https://huggingface.co/Wan-AI/Wan2.6-R2V-14B)
- [LivePortrait](https://github.com/KwaiVGI/LivePortrait)
- InsightFace models (downloaded automatically on first use where applicable)

### Start development

```bash
# Start backend, frontend, worker, and Redis together
make dev
```

Or individually:

```bash
make backend
make frontend
make worker
make redis
```

Open:

- Frontend: http://localhost:3000
- Backend docs: http://localhost:8000/docs

### Docker

```bash
make docker-build
make docker-up
make docker-logs
```

## Realtime configuration

Realtime sessions are created with a `RealtimeConfig` payload. The backend now supports high-resolution and adaptive-control fields such as:

- `input_resolution`
- `output_resolution`
- `target_fps`
- `jpeg_quality`
- `jpeg_subsampling`
- `binary_transport`
- `full_frame_inference`
- `tile_size`
- `tile_overlap`
- `max_inflight_frames`
- `allow_frame_drop`
- `adaptive_quality`
- `adaptive_latency_budget_ms`
- `adaptive_jpeg_step`
- `adaptive_min_jpeg_quality`
- `adaptive_cooldown_frames`
- `adaptive_tile_size`
- `adaptive_min_tile_size`
- `adaptive_fps_step`
- `adaptive_min_target_fps`

### Example realtime session payload

```json
{
  "reference_image_id": "abc123",
  "mode": "liveportrait",
  "target_fps": 24,
  "face_only": true,
  "smoothing": 0.5,
  "enhance_face": true,
  "input_resolution": [1920, 1080],
  "output_resolution": [1920, 1080],
  "jpeg_quality": 92,
  "jpeg_subsampling": "420",
  "binary_transport": true,
  "full_frame_inference": true,
  "tile_size": null,
  "tile_overlap": 64,
  "max_inflight_frames": 1,
  "allow_frame_drop": true,
  "adaptive_quality": true,
  "adaptive_latency_budget_ms": 42,
  "adaptive_jpeg_step": 5,
  "adaptive_min_jpeg_quality": 75,
  "adaptive_cooldown_frames": 24,
  "adaptive_tile_size": 1024,
  "adaptive_min_tile_size": 512,
  "adaptive_fps_step": 6,
  "adaptive_min_target_fps": 15
}
```

## API reference

### Upload

```text
POST /api/upload/image
POST /api/upload/video
```

### Generation

```text
POST /api/generate
POST /api/generate/preview
GET  /api/generate/modes
```

### Jobs

```text
GET    /api/jobs
GET    /api/jobs/{id}
GET    /api/jobs/{id}/progress
POST   /api/jobs/{id}/cancel
DELETE /api/jobs/{id}
```

### Realtime REST endpoints

```text
POST   /api/realtime/session
GET    /api/realtime/session/{session_id}
GET    /api/realtime/session/{session_id}/metrics
DELETE /api/realtime/session/{session_id}
GET    /api/realtime/check-compatibility
GET    /api/realtime/workers
```

### WebSocket endpoints

```text
WS /ws/jobs/{job_id}
WS /ws/realtime/{session_id}
WS /ws/status
```

### Realtime WebSocket protocol

- Realtime frame transport is now **binary JPEG only**
- JSON text messages are reserved for:
  - `connected`
  - `ping` / `pong`
  - `adaptive_update`
  - `error`
  - `stop`

## Frontend realtime tools

The realtime page includes:

- resolution presets (720p / 1080p / 4K)
- target FPS selection
- JPEG quality slider
- compatibility check + recommended preset
- optional debug panel showing:
  - session latency metrics
  - adaptive events
  - effective quality settings
  - worker telemetry
  - shared-memory vs inline transport counts

## Backend realtime services

Key realtime modules:

- `backend/services/realtime/worker_pool.py`
  - worker lifecycle
  - per-session worker affinity
  - worker telemetry
- `backend/services/realtime/shared_memory.py`
  - shared-memory payload transport for large frames
- `backend/services/realtime/metrics.py`
  - session metrics aggregation
- `backend/services/realtime/adaptive.py`
  - adaptive quality controller
- `backend/services/realtime/pipeline.py`
  - per-frame decode / process / encode pipeline
- `backend/services/realtime/tiled_inference.py`
  - tiled processing helpers

## Project structure

```text
vfx-motion-capture/
├── backend/
│   ├── api/                  # FastAPI routes and WebSocket handlers
│   ├── core/                 # Configuration, models, exceptions
│   ├── services/
│   │   ├── inference/        # Face swap, portrait, Wan inference services
│   │   └── realtime/         # Worker pool, metrics, adaptive policy, transport helpers
│   ├── workers/              # Celery async tasks
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── pages/
│   │   └── services/
├── models/                   # Model weights (git-ignored)
├── scripts/                  # Setup and model download helpers
├── docker-compose.yml
├── Makefile
└── README.md
```

## Environment and config

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Common settings:

```bash
# Runtime
DEVICE=auto
ENABLE_FP16=true
ENABLE_XFORMERS=true

# App
HOST=0.0.0.0
PORT=8000

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
```

Backend dependencies live in:

- `backend/requirements.txt`

Notable realtime dependency:

- `PyTurboJPEG` for accelerated JPEG encode/decode when available

## Known limitations

- Realtime sessions and metrics are currently stored **in process memory**
- The built-in LivePortrait path is a real landmark-driven fallback renderer, but not a full external LivePortrait neural pipeline unless that package is installed and integrated successfully
- Frontend type-check currently still has unrelated existing errors in `frontend/src/pages/index.tsx`

## Troubleshooting

### Realtime FPS is low

- Open the realtime debug panel and inspect:
  - dropped frames
  - worker saturation
  - shared-memory usage
  - adaptive events
- Lower capture/output resolution to 1080p before trying 4K
- Reduce target FPS or JPEG quality
- Check whether the backend has already stepped down settings adaptively

### Adaptive quality is triggering too often

Tune these fields in realtime session config or backend settings:

- `adaptive_latency_budget_ms`
- `adaptive_jpeg_step`
- `adaptive_fps_step`
- `adaptive_cooldown_frames`
- `adaptive_tile_size`

### Shared memory is not being used

Check:

- `realtime_use_shared_memory`
- `realtime_shared_memory_threshold_bytes`

Large JPEG payloads above the threshold should move through shared memory.

### Apple Silicon performance tuning

For local Apple Silicon testing:

- start with 1080p / 24 FPS
- use the recommended preset on the realtime page
- keep the debug panel open and watch:
  - worker latency
  - total latency
  - worker saturation
  - adaptive events

### Docker / Redis issues

- Start Redis with `make redis` or `make docker-up`
- Check logs with `make docker-logs`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run backend tests: `make test`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Wan Video](https://github.com/Wan-Video/Wan2.1)
- [LivePortrait](https://github.com/KwaiVGI/LivePortrait)
- [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)
- [InsightFace](https://github.com/deepinsight/insightface)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
