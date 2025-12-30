# VFX Motion Capture - Real-Time Person Replacement System

## Project Overview

A real-time VFX motion capture application that enables:
1. **Image-to-Video Mode**: Replace a person in an input video with a person from a reference image
2. **Real-Time Camera Mode**: Use phone/laptop camera to replace a person in a video in real-time

## Technology Stack

### Core AI Models

| Model | Purpose | VRAM | Speed |
|-------|---------|------|-------|
| **Wan 2.6 R2V** | Reference-to-Video generation with identity preservation | 24GB+ | ~2-4 min/15s video |
| **Wan 2.1/2.2 VACE** | Pose transfer, motion control, video-to-video | 12-24GB | ~30s-2min/5s video |
| **LivePortrait** | Real-time face animation (12.8ms/frame on RTX 4090) | 4-8GB | Real-time |
| **Deep-Live-Cam** | Real-time face swap with single image | 4-8GB | Real-time |

### Backend Infrastructure

- **ComfyUI**: Node-based AI workflow engine with Python API
- **ComfyStream**: Real-time video processing extension for ComfyUI
- **FastAPI/Flask**: REST API server for job management
- **Redis/Celery**: Job queue for async video processing
- **WebSocket**: Real-time status updates and streaming

### Frontend

- **React/Next.js** or **Gradio**: Web UI framework
- **WebRTC**: Camera access and real-time streaming
- **FFmpeg.wasm**: Client-side video format handling

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (Web UI)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Image Upload │  │ Video Upload │  │ Camera Feed  │  │ Prompt Input │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                              │                │                              │
│                              ▼                ▼                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                        WebRTC / WebSocket                           │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND (FastAPI)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Job Queue   │  │ File Handler │  │ Prompt Parser│  │ Model Router │    │
│  │   (Redis)    │  │   (FFmpeg)   │  │   (LLM)      │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│   High-Quality Mode     │ │   Fast Turbo Mode       │ │   Real-Time Mode        │
├─────────────────────────┤ ├─────────────────────────┤ ├─────────────────────────┤
│ • Wan 2.6 R2V           │ │ • Wan 2.1 VACE Turbo    │ │ • LivePortrait          │
│ • 15s 1080p video       │ │ • Wan 2.2 Animate       │ │ • Deep-Live-Cam         │
│ • Full identity + voice │ │ • 5s 720p video         │ │ • ComfyStream           │
│ • Multi-shot support    │ │ • Quick iterations      │ │ • <50ms latency         │
└─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ComfyUI Backend                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Pose Extract │  │ Face Detect  │  │ Motion Xfer  │  │ Video Render │    │
│  │  (DWPose)    │  │ (InsightFace)│  │   (VACE)     │  │   (FFmpeg)   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Project Setup & Infrastructure

#### 1.1 Project Structure
```
vfx-motion-capture/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app
│   │   ├── routes/
│   │   │   ├── upload.py        # File upload endpoints
│   │   │   ├── generate.py      # Video generation endpoints
│   │   │   ├── realtime.py      # Real-time streaming endpoints
│   │   │   └── jobs.py          # Job status endpoints
│   │   └── websocket.py         # WebSocket handlers
│   ├── core/
│   │   ├── config.py            # Configuration management
│   │   ├── models.py            # Pydantic models
│   │   └── exceptions.py        # Custom exceptions
│   ├── services/
│   │   ├── comfyui_client.py    # ComfyUI API wrapper
│   │   ├── video_processor.py   # FFmpeg video handling
│   │   ├── prompt_parser.py     # Natural language prompt parsing
│   │   ├── pose_extractor.py    # DWPose extraction
│   │   └── face_detector.py     # Face detection service
│   ├── workers/
│   │   ├── celery_app.py        # Celery configuration
│   │   ├── video_tasks.py       # Async video generation tasks
│   │   └── realtime_tasks.py    # Real-time processing tasks
│   ├── comfyui_workflows/
│   │   ├── wan_vace_pose_transfer.json
│   │   ├── wan_r2v_character.json
│   │   ├── liveportrait_animate.json
│   │   └── deep_live_cam.json
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUploader.tsx
│   │   │   ├── VideoUploader.tsx
│   │   │   ├── CameraFeed.tsx
│   │   │   ├── PromptInput.tsx
│   │   │   ├── VideoPlayer.tsx
│   │   │   ├── GenerateButton.tsx
│   │   │   └── ProgressBar.tsx
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts
│   │   │   ├── useWebRTC.ts
│   │   │   └── useVideoGeneration.ts
│   │   ├── services/
│   │   │   └── api.ts
│   │   ├── pages/
│   │   │   ├── index.tsx        # Main app page
│   │   │   └── realtime.tsx     # Real-time mode page
│   │   └── App.tsx
│   ├── package.json
│   └── tailwind.config.js
├── comfyui/
│   ├── docker-compose.yml       # ComfyUI containerized setup
│   └── custom_nodes/            # Custom nodes if needed
├── models/                      # Model weights (git-ignored)
│   ├── wan2.6/
│   ├── wan2.1_vace/
│   ├── liveportrait/
│   └── insightface/
├── docker-compose.yml           # Full stack orchestration
├── Makefile                     # Common commands
└── README.md
```

#### 1.2 Dependencies
```
# Backend (Python 3.10+)
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
websockets>=12.0
celery>=5.3.4
redis>=5.0.1
ffmpeg-python>=0.2.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
httpx>=0.25.0
pydantic>=2.5.0
torch>=2.1.0
torchvision>=0.16.0

# Frontend (Node.js 18+)
next@14
react@18
tailwindcss@3
socket.io-client
```

---

### Phase 2: Core Backend Services

#### 2.1 ComfyUI Integration Service
- Connect to ComfyUI via WebSocket API
- Load and execute workflow JSON files
- Handle queue management and progress tracking
- Support workflow parameter injection

#### 2.2 Video Processing Service
- Input format handling: .mov, .mp4, .mpeg, .avi, .webm
- Frame extraction and resampling
- Audio separation and reattachment
- Output encoding with hardware acceleration

#### 2.3 Prompt Parser Service
- Parse natural language prompts like:
  - "Replace person in video with person in reference image"
  - "Make the subject dance like in the reference video"
  - "Transfer the motion to my character"
- Map to appropriate model/workflow selection
- Extract parameters (style, intensity, etc.)

#### 2.4 Pose Extraction Service
- DWPose for body pose extraction
- MediaPipe as fallback
- Generate pose sequences from input videos

---

### Phase 3: AI Model Integration

#### 3.1 Wan 2.6 R2V (Reference-to-Video)
**Use Case**: High-quality character insertion with identity preservation
```python
# Workflow: Upload reference image/video → Generate new scenes
{
    "reference_image": "path/to/character.jpg",
    "prompt": "Character walking through a forest",
    "duration": 15,  # seconds
    "resolution": "1080p"
}
```

#### 3.2 Wan 2.1/2.2 VACE Pose Transfer
**Use Case**: Transfer motion from source video to reference character
```python
# Workflow: Reference image + Motion video → Character performing motion
{
    "reference_image": "path/to/character.jpg",
    "motion_video": "path/to/dance.mp4",
    "mode": "pose_transfer",
    "strength": 0.85
}
```

#### 3.3 LivePortrait Real-Time
**Use Case**: Real-time face animation from camera
```python
# Workflow: Source image + Live camera → Animated character
{
    "source_image": "path/to/character.jpg",
    "driving_source": "webcam",
    "fps": 30,
    "smoothing": 0.5
}
```

#### 3.4 Deep-Live-Cam
**Use Case**: Real-time face swap in video
```python
# Workflow: Source face + Target video → Face-swapped output
{
    "source_face": "path/to/face.jpg",
    "target_video": "path/to/video.mp4",
    "enhance_face": True
}
```

---

### Phase 4: Frontend Implementation

#### 4.1 Main Interface Layout
```
┌─────────────────────────────────────────────────────────────────┐
│  VFX Motion Capture                              [Mode: v]      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐     ┌─────────────────┐                   │
│  │                 │     │                 │                   │
│  │  Reference      │     │  Input Video    │                   │
│  │  Image/Video    │     │  or Camera      │                   │
│  │                 │     │                 │                   │
│  │  [Upload]       │     │  [Upload/Start] │                   │
│  └─────────────────┘     └─────────────────┘                   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Prompt: "Replace person with reference character..."    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  [Quality: Standard v] [Duration: Auto v]  [🚀 Generate]       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │                    Output Preview                       │   │
│  │                                                         │   │
│  │                    [▶ Play] [⬇ Download]                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Progress: [████████████░░░░░░░░] 60% - Generating frames...   │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.2 Real-Time Mode Interface
```
┌─────────────────────────────────────────────────────────────────┐
│  VFX Motion Capture - Real-Time Mode              [⚙ Settings] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌────────────────────────────────────────┐  │
│  │              │  │                                        │  │
│  │  Reference   │  │                                        │  │
│  │  Character   │  │           Live Output                  │  │
│  │              │  │                                        │  │
│  │  [Change]    │  │         (Your camera feed              │  │
│  └──────────────┘  │          with character)               │  │
│                    │                                        │  │
│  Camera:           │                                        │  │
│  [Webcam v]        │                                        │  │
│                    └────────────────────────────────────────┘  │
│  Mode:                                                         │
│  ○ Face Only        FPS: 28 │ Latency: 45ms │ [🔴 Recording]  │
│  ● Full Body                                                   │
│  ○ Motion Transfer                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Phase 5: Real-Time Pipeline

#### 5.1 Camera Capture (WebRTC)
```javascript
// Browser captures frames at 30fps
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    // Send frames to backend via WebSocket
  });
```

#### 5.2 Processing Pipeline
```
Camera Frame → Pose Extraction → LivePortrait/DeepLiveCam → Output Frame
     ↓              ↓                    ↓                      ↓
   30fps          ~10ms               ~15ms                  ~5ms
                          Total: ~30-50ms latency
```

#### 5.3 ComfyStream Integration
- Use ComfyStream for running ComfyUI workflows on live video
- Maintain frame buffer for smooth playback
- Handle dropped frames gracefully

---

### Phase 6: API Endpoints

#### 6.1 REST Endpoints
```
POST /api/upload/image          # Upload reference image
POST /api/upload/video          # Upload input video
POST /api/generate              # Start generation job
GET  /api/jobs/{job_id}         # Get job status
GET  /api/jobs/{job_id}/result  # Download result
DELETE /api/jobs/{job_id}       # Cancel job
```

#### 6.2 WebSocket Endpoints
```
WS /ws/generate/{job_id}        # Real-time generation progress
WS /ws/realtime                 # Real-time camera processing
```

---

### Phase 7: Model Download & Setup

#### 7.1 Required Model Downloads
```bash
# Wan 2.6 (for R2V)
huggingface-cli download Wan-AI/Wan2.6-R2V --local-dir models/wan2.6

# Wan 2.1 VACE (for pose transfer)
huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir models/wan2.1_vace

# LivePortrait
git clone https://github.com/KwaiVGI/LivePortrait models/liveportrait

# InsightFace (for face detection)
pip install insightface
# Models auto-download on first use

# DWPose (for pose extraction)
# Included in ComfyUI controlnet_aux
```

#### 7.2 Hardware Requirements
| Mode | Minimum GPU | Recommended GPU | VRAM |
|------|-------------|-----------------|------|
| Real-time (LivePortrait) | RTX 3060 | RTX 4070+ | 8GB |
| Fast (VACE Turbo) | RTX 3080 | RTX 4080+ | 12GB |
| High-Quality (Wan 2.6) | RTX 4090 | A100/H100 | 24GB+ |

---

## File Format Support

### Input Formats
- **Images**: .jpg, .jpeg, .png, .webp, .bmp
- **Videos**: .mp4, .mov, .mpeg, .avi, .webm, .mkv

### Output Formats
- **Video**: .mp4 (H.264), .webm (VP9)
- **GIF**: For short clips

---

## Prompt Examples

| Prompt | Action |
|--------|--------|
| "Replace person in video with reference image" | VACE pose transfer |
| "Make my character dance like in the video" | VACE motion transfer |
| "Put me in this scene" | Wan 2.6 R2V |
| "Animate this portrait with my expressions" | LivePortrait |
| "Swap my face with the character" | Deep-Live-Cam |

---

## Implementation Order

1. **Week 1-2**: Project setup, ComfyUI integration, basic file upload
2. **Week 2-3**: Wan VACE pose transfer workflow
3. **Week 3-4**: Frontend UI, WebSocket progress tracking
4. **Week 4-5**: LivePortrait real-time integration
5. **Week 5-6**: Prompt parsing, model routing
6. **Week 6-7**: Real-time camera mode with WebRTC
7. **Week 7-8**: Polish, error handling, documentation

---

## References

- [Wan 2.6 Official](https://wan2.video/wan2.6)
- [Wan 2.1 GitHub](https://github.com/Wan-Video/Wan2.1)
- [LivePortrait GitHub](https://github.com/KwaiVGI/LivePortrait)
- [Deep-Live-Cam GitHub](https://github.com/hacksider/Deep-Live-Cam)
- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
- [ComfyStream Blog](https://blog.livepeer.org/building-real-time-ai-video-effects-with-comfystream/)
- [Wan VACE ComfyUI Tutorial](https://stable-diffusion-art.com/wan-vace-ref/)
