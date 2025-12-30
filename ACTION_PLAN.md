# VFX Motion Capture - Action Plan

## Table of Contents
1. [Placeholder Code & Bugs Found](#1-placeholder-code--bugs-found)
2. [Action Plan to Fix Issues](#2-action-plan-to-fix-issues)
3. [MacOS Apple Silicon Optimization Plan](#3-macos-apple-silicon-optimization-plan)
4. [Non-ComfyUI Implementation Plan](#4-non-comfyui-implementation-plan)

---

## 1. Placeholder Code & Bugs Found

### 1.1 Critical Placeholder Code

#### 1.1.1 Real-time Processor Initialization (`backend/api/websocket.py`)

**Location**: Lines 294-322

```python
if mode == GenerationMode.LIVEPORTRAIT:
    # Initialize LivePortrait
    try:
        # Note: This is a placeholder - actual implementation would load
        # the LivePortrait model
        logger.info("Initializing LivePortrait processor")
        # from liveportrait import LivePortraitPipeline
        # processor["model"] = LivePortraitPipeline()
        # processor["model"].prepare(reference_image)
    except Exception as e:
        logger.warning(f"Failed to initialize LivePortrait: {e}")
```

**Issue**: LivePortrait model is never actually loaded. The real-time processing is completely non-functional.

---

#### 1.1.2 Real-time Frame Processing (`backend/api/websocket.py`)

**Location**: Lines 347-359

```python
if mode == GenerationMode.LIVEPORTRAIT:
    # Apply LivePortrait transformation
    if processor.get("model"):
        # result = processor["model"].animate(frame_rgb)
        pass
    else:
        # Fallback: simple color adjustment to show processing
        result = cv2.addWeighted(...)  # Just a blend, not actual transformation
```

**Issue**: The actual LivePortrait animation call is commented out. Falls back to a simple image blend.

---

#### 1.1.3 Face Swap Processing (`backend/api/websocket.py`)

**Location**: Lines 361-379

```python
elif mode == GenerationMode.DEEP_LIVE_CAM:
    # In a full implementation, we would:
    # 1. Align the source face to the target position
    # 2. Blend the faces together
    # 3. Apply color correction
    # For now, just overlay the reference
    source_face = processor.get("source_face")
    if source_face:
        # Draw indicator that processing is happening
        result = face_detector.draw_faces(frame_rgb, faces)
```

**Issue**: No actual face swapping occurs. Only draws bounding boxes around detected faces.

---

#### 1.1.4 Motion Transfer is Clone of Pose Transfer (`backend/workers/video_tasks.py`)

**Location**: Lines 279-290

```python
async def generate_vace_motion_transfer(...) -> Dict[str, Any]:
    """Generate video using Wan VACE motion transfer."""
    # Similar to pose transfer but uses motion control instead
    return await generate_vace_pose_transfer(
        job_id, ref_image_path, input_video_path, request
    )
```

**Issue**: Motion transfer simply calls pose transfer. No differentiated implementation.

---

#### 1.1.5 DWPose Extraction Fallback (`backend/services/pose_extractor.py`)

**Location**: Lines 135-148, 253-270

```python
async def _init_dwpose(self) -> None:
    """Initialize DWPose (requires controlnet_aux)."""
    try:
        from controlnet_aux import DWposeDetector
        self._model = DWposeDetector()
    except ImportError:
        logger.warning("DWPose not available, falling back to MediaPipe")
        self.backend = "mediapipe"
        await self._init_mediapipe()
```

**Issue**: DWPose integration is incomplete. The `_extract_dwpose` method has placeholder code that assumes a specific result format that may not match the actual DWPose API.

---

### 1.2 Bugs Found

#### 1.2.1 Secret Key Security Issue (`backend/core/config.py`)

**Location**: Line 121

```python
secret_key: str = "your-secret-key-change-in-production"
```

**Bug**: Hardcoded default secret key is insecure. Should fail startup if not configured.

---

#### 1.2.2 Missing Exception Import Shadow (`backend/core/exceptions.py`)

**Location**: Line 109

```python
class FileNotFoundError(VFXException):
```

**Bug**: This shadows Python's built-in `FileNotFoundError`. Should be renamed to `FileNotFoundVFXError` or similar.

---

#### 1.2.3 Memory Leak in Singleton Patterns (Multiple Files)

**Location**: All service files with `_client`, `_processor`, `_extractor` singletons

**Bug**: Singletons are never cleaned up on shutdown. In production, this can cause memory leaks and resource exhaustion. No lifecycle management.

---

#### 1.2.4 Async Generator Issue (`backend/services/video_processor.py`)

**Location**: Lines 200-251 - `extract_frames_as_arrays`

```python
async def extract_frames_as_arrays(...) -> AsyncIterator[Tuple[int, np.ndarray]]:
```

**Bug**: This function yields from a synchronous `cv2.VideoCapture` loop. The `await asyncio.sleep(0)` is a band-aid. Should use proper async video reading.

---

#### 1.2.5 Exception Handling Shadow (`backend/api/routes/generate.py`)

**Location**: Lines 69-73

```python
except FileNotFoundError:
    raise HTTPException(...)
```

**Bug**: This catches Python's built-in `FileNotFoundError`, not the custom `backend.core.exceptions.FileNotFoundError`. The import is correct but due to shadowing, behavior is ambiguous.

---

#### 1.2.6 In-Memory Storage for Production (`backend/services/file_manager.py`, `backend/services/job_manager.py`)

**Location**: Lines 64-65 (file_manager), Lines 105-108 (job_manager)

```python
# In-memory file index (in production, use a database)
self._files: Dict[str, FileInfo] = {}
```

**Bug**: File and job data is lost on restart. Comments acknowledge this but no database implementation exists.

---

#### 1.2.7 Potential Race Condition in Job Manager (`backend/services/job_manager.py`)

**Location**: Lines 346-363 - `_start_next_queued`

**Bug**: The method checks active count and promotes jobs without holding the lock the entire time. Race conditions can occur under high concurrency.

---

#### 1.2.8 Hardcoded NVIDIA GPU Dependency (`docker-compose.yml`)

**Location**: Lines 39-45, 69-75, 100-106

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

**Bug**: Makes the application impossible to run without NVIDIA GPUs.

---

#### 1.2.9 Missing Error Handling in ComfyUI Workflow Parameter Injection (`backend/services/comfyui_client.py`)

**Location**: Lines 196-224

**Bug**: No validation that required parameters exist in workflow. Silent failures if node structure doesn't match expectations.

---

#### 1.2.10 WebSocket Session Cleanup (`backend/api/routes/realtime.py`)

**Location**: Lines 70-76

```python
# Store session config (in production, use Redis or similar)
from backend.api.websocket import realtime_sessions
realtime_sessions[session_id] = {...}
```

**Bug**: Sessions are never cleaned up if client disconnects unexpectedly. Memory leak over time.

---

### 1.3 Missing Functionality

1. **No Health Endpoint Implementation**: Referenced in Dockerfile healthcheck but `/health` endpoint isn't defined
2. **No File Download Endpoints**: `files.py` is referenced but missing actual implementation
3. **No Job Retry Endpoint**: API client references `/jobs/{id}/retry` but backend doesn't implement it
4. **No Progress Preview Images**: `preview_url` in JobProgress is always None
5. **Missing Celery App Configuration**: `celery_app.py` mentioned but likely incomplete
6. **No Rate Limiting**: `RateLimitError` exists but no rate limiting middleware
7. **No Authentication**: No auth middleware despite security config fields

---

## 2. Action Plan to Fix Issues

### Phase 1: Critical Fixes (Week 1)

#### 1.1 Fix Security Issues
- [ ] **Generate secure secret key at startup**
  ```python
  # config.py
  import secrets
  secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
  
  @field_validator('secret_key')
  def validate_secret(cls, v):
      if v == "your-secret-key-change-in-production":
          raise ValueError("Must set SECRET_KEY environment variable")
      return v
  ```

#### 1.2 Rename Shadowed Exception
- [ ] Rename `FileNotFoundError` to `VFXFileNotFoundError` in `exceptions.py`
- [ ] Update all imports and usages across codebase

#### 1.3 Add Missing Health Endpoint
- [ ] Create `/health` endpoint in `main.py`:
  ```python
  @app.get("/health")
  async def health_check():
      return {
          "status": "healthy",
          "version": settings.app_version,
          "comfyui": await check_comfyui_health()
      }
  ```

### Phase 2: Implement Missing Features (Weeks 2-3)

#### 2.1 Implement Real LivePortrait Integration
- [ ] Install LivePortrait dependencies
- [ ] Create `backend/services/liveportrait_service.py`:
  ```python
  class LivePortraitService:
      async def initialize(self, source_image: np.ndarray) -> None
      async def animate(self, driving_frame: np.ndarray) -> np.ndarray
      async def close(self) -> None
  ```
- [ ] Update `websocket.py` to use actual implementation

#### 2.2 Implement Real Face Swap
- [ ] Integrate `insightface` swapper models
- [ ] Create `backend/services/face_swap_service.py`:
  ```python
  class FaceSwapService:
      async def swap_face(
          self, 
          source_face: DetectedFace, 
          target_frame: np.ndarray
      ) -> np.ndarray
  ```

#### 2.3 Implement Motion Transfer Differently from Pose Transfer
- [ ] Create distinct workflow for motion transfer
- [ ] Use different VACE control signals

#### 2.4 Add Database Persistence
- [ ] Implement SQLAlchemy models for jobs and files
- [ ] Add migration support with Alembic
- [ ] Create database-backed `JobRepository` and `FileRepository`

### Phase 3: Production Hardening (Week 4)

#### 3.1 Add Lifecycle Management
- [ ] Implement proper shutdown handlers:
  ```python
  @app.on_event("shutdown")
  async def shutdown():
      await get_comfyui_client().close()
      get_pose_extractor().close()
      get_face_detector().close()
  ```

#### 3.2 Add Session Cleanup
- [ ] Implement periodic cleanup task for realtime sessions
- [ ] Add TTL to sessions with automatic expiration

#### 3.3 Add Rate Limiting
- [ ] Add `slowapi` or similar rate limiting middleware
- [ ] Configure per-IP and per-user limits

#### 3.4 Add Missing API Endpoints
- [ ] Implement `/jobs/{id}/retry`
- [ ] Implement file download endpoints
- [ ] Implement progress preview image generation

---

## 3. MacOS Apple Silicon Optimization Plan

### 3.1 Overview

Apple Silicon (M1/M2/M3) uses MPS (Metal Performance Shaders) instead of CUDA. This requires significant changes across the codebase.

### 3.2 Dependency Changes

#### 3.2.1 Update `requirements.txt`
```python
# Replace CUDA-specific packages
# onnxruntime-gpu>=1.16.0  # REMOVE
onnxruntime>=1.16.0  # CPU/CoreML fallback

# Add MPS-compatible packages
coremltools>=7.0  # For CoreML conversion
```

#### 3.2.2 PyTorch MPS Support
```python
# Already supports MPS, no change needed but version check:
torch>=2.1.0  # MPS support stable from 2.0+
```

### 3.3 Code Changes

#### 3.3.1 Device Detection (`backend/core/config.py`)

Add auto-detection for Apple Silicon:

```python
import platform
import torch

def get_optimal_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class Settings(BaseSettings):
    # Change default device detection
    device: str = Field(default_factory=get_optimal_device)
```

#### 3.3.2 Update Face Detector (`backend/services/face_detector.py`)

```python
async def initialize(self) -> None:
    import platform
    
    # Determine providers based on device
    if self.device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif self.device == "mps":
        # ONNX Runtime on Mac - use CoreML or CPU
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    
    # For InsightFace on Apple Silicon
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        # InsightFace needs special handling on M1/M2
        import os
        os.environ["OPENBLAS_NUM_THREADS"] = "4"
        os.environ["OMP_NUM_THREADS"] = "4"
```

#### 3.3.3 Update Real-time Compatibility Check (`backend/api/routes/realtime.py`)

```python
@router.get("/realtime/check-compatibility")
async def check_compatibility():
    import torch
    import platform
    
    gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    gpu_type = "none"
    gpu_name = None
    gpu_memory = None
    
    if torch.cuda.is_available():
        gpu_type = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif torch.backends.mps.is_available():
        gpu_type = "mps"
        gpu_name = f"Apple {platform.processor()} GPU"
        # MPS doesn't expose memory info directly
        # Estimate based on chip
        import subprocess
        result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True)
        total_ram = int(result.stdout) / (1024**3)
        # Apple Silicon shares memory - estimate GPU portion
        gpu_memory = total_ram * 0.6  # Rough estimate
    
    # Estimate capability
    capability = "none"
    estimated_fps = 0
    
    if gpu_type == "mps":
        # Apple Silicon capabilities by chip type
        chip = platform.processor()
        if "M3" in str(chip) or "M2" in str(chip):
            capability = "good"
            estimated_fps = 25
        elif "M1" in str(chip):
            capability = "moderate"
            estimated_fps = 15
    elif gpu_type == "cuda":
        # Existing CUDA logic...
        pass
    
    return {
        "gpu_available": gpu_available,
        "gpu_type": gpu_type,
        "gpu_name": gpu_name,
        "gpu_memory_gb": round(gpu_memory, 1) if gpu_memory else None,
        "capability": capability,
        "estimated_fps": estimated_fps,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
    }
```

#### 3.3.4 Create MPS-Compatible Tensor Operations

Add `backend/services/device_utils.py`:

```python
"""Device-agnostic tensor utilities."""
import torch
from typing import Literal

DeviceType = Literal["cuda", "mps", "cpu"]

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def to_device(tensor: torch.Tensor, device: DeviceType = None) -> torch.Tensor:
    """Move tensor to specified device with fallbacks."""
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)
    
    # MPS doesn't support all dtypes
    if device.type == "mps":
        # Convert unsupported dtypes
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        elif tensor.dtype == torch.int64:
            tensor = tensor.int()
    
    return tensor.to(device)

def supports_half_precision() -> bool:
    """Check if device supports FP16."""
    device = get_device()
    if device.type == "cuda":
        return True
    elif device.type == "mps":
        # MPS has limited FP16 support
        return True
    return False
```

### 3.4 Docker Configuration

#### 3.4.1 Create macOS-compatible Docker Compose

Create `docker-compose.macos.yml`:

```yaml
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
    depends_on:
      - backend
    platform: linux/arm64  # For M1/M2/M3

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.macos
    platform: linux/arm64
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - DEVICE=cpu  # MPS not available in Docker
      - COMFYUI_HOST=comfyui
      - COMFYUI_PORT=8188
    volumes:
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
      - ./models:/app/models
    depends_on:
      - redis
    # No GPU reservation needed

  redis:
    image: redis:7-alpine
    platform: linux/arm64
    ports:
      - "6379:6379"
```

#### 3.4.2 Create macOS Dockerfile

Create `backend/Dockerfile.macos`:

```dockerfile
FROM python:3.11-slim

# Install system dependencies (ARM64 compatible)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements (no CUDA)
COPY requirements.macos.txt .
RUN pip install --no-cache-dir -r requirements.macos.txt

COPY . .
RUN mkdir -p uploads outputs temp models data

EXPOSE 8000

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.5 Native macOS (Non-Docker) Setup

Create `scripts/setup_macos.sh`:

```bash
#!/bin/bash
# macOS Apple Silicon Setup Script

set -e

echo "VFX Motion Capture - macOS Apple Silicon Setup"
echo "=============================================="

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "Warning: This script is optimized for Apple Silicon (M1/M2/M3)"
fi

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install dependencies
echo "Installing system dependencies..."
brew install python@3.11 ffmpeg redis

# Create virtual environment
echo "Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch for MPS
echo "Installing PyTorch with MPS support..."
pip install --upgrade pip
pip install torch torchvision torchaudio

# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Install remaining dependencies
echo "Installing Python dependencies..."
pip install -r backend/requirements.macos.txt

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend && npm install && cd ..

echo ""
echo "Setup complete!"
echo ""
echo "To start:"
echo "  1. Start Redis: brew services start redis"
echo "  2. Start backend: source venv/bin/activate && python -m uvicorn backend.api.main:app"
echo "  3. Start frontend: cd frontend && npm run dev"
```

### 3.6 Model Optimization for Apple Silicon

#### 3.6.1 CoreML Conversion Script

Create `scripts/convert_models_coreml.py`:

```python
"""Convert models to CoreML for Apple Silicon optimization."""
import coremltools as ct
import torch
from pathlib import Path

def convert_to_coreml(
    model: torch.nn.Module,
    input_shape: tuple,
    output_path: Path,
    model_name: str
):
    """Convert PyTorch model to CoreML."""
    model.eval()
    
    # Trace the model
    example_input = torch.randn(*input_shape)
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=input_shape)],
        compute_units=ct.ComputeUnit.ALL  # Use ANE + GPU + CPU
    )
    
    # Optimize for Apple Neural Engine
    mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
        mlmodel, 
        nbits=16  # FP16 for ANE
    )
    
    output_file = output_path / f"{model_name}.mlpackage"
    mlmodel.save(str(output_file))
    print(f"Saved CoreML model: {output_file}")

if __name__ == "__main__":
    # Convert face detection model
    # convert_to_coreml(face_model, (1, 3, 640, 640), Path("models/coreml"), "face_detector")
    pass
```

### 3.7 Performance Benchmarks to Implement

Create `scripts/benchmark_apple_silicon.py`:

```python
"""Benchmark script for Apple Silicon performance."""
import time
import torch
import numpy as np

def benchmark_mps():
    """Benchmark MPS vs CPU performance."""
    device_mps = torch.device("mps") if torch.backends.mps.is_available() else None
    device_cpu = torch.device("cpu")
    
    # Test matrix operations
    sizes = [(512, 512), (1024, 1024), (2048, 2048)]
    
    print("Matrix Multiplication Benchmark")
    print("=" * 50)
    
    for size in sizes:
        # CPU
        a_cpu = torch.randn(size, device=device_cpu)
        b_cpu = torch.randn(size, device=device_cpu)
        
        start = time.perf_counter()
        for _ in range(100):
            _ = torch.mm(a_cpu, b_cpu)
        cpu_time = (time.perf_counter() - start) / 100
        
        # MPS
        if device_mps:
            a_mps = a_cpu.to(device_mps)
            b_mps = b_cpu.to(device_mps)
            
            # Warmup
            for _ in range(10):
                _ = torch.mm(a_mps, b_mps)
            torch.mps.synchronize()
            
            start = time.perf_counter()
            for _ in range(100):
                _ = torch.mm(a_mps, b_mps)
            torch.mps.synchronize()
            mps_time = (time.perf_counter() - start) / 100
            
            speedup = cpu_time / mps_time
            print(f"Size {size}: CPU={cpu_time*1000:.2f}ms, MPS={mps_time*1000:.2f}ms, Speedup={speedup:.2f}x")
        else:
            print(f"Size {size}: CPU={cpu_time*1000:.2f}ms, MPS=N/A")

if __name__ == "__main__":
    benchmark_mps()
```

---

## 4. Non-ComfyUI Implementation Plan

### 4.1 Overview

Removing ComfyUI dependency requires implementing direct model inference. This is a significant undertaking but provides more control and potentially better performance.

### 4.2 Architecture Change

```
Current:
  Backend → ComfyUI (WebSocket) → Model Inference → Output

Target:
  Backend → Direct Python Inference → Output
```

### 4.3 New Service Structure

```
backend/
├── services/
│   ├── inference/           # NEW: Direct inference services
│   │   ├── __init__.py
│   │   ├── base.py          # Base inference class
│   │   ├── wan_vace.py      # Wan VACE inference
│   │   ├── wan_r2v.py       # Wan R2V inference
│   │   ├── liveportrait.py  # LivePortrait inference
│   │   ├── face_swap.py     # Face swap inference
│   │   └── model_loader.py  # Model loading utilities
│   ├── comfyui_client.py    # DEPRECATED - keep for backward compat
```

### 4.4 Implementation Steps

#### Phase 1: Create Base Infrastructure (Week 1)

##### 4.4.1 Base Inference Class

Create `backend/services/inference/base.py`:

```python
"""Base class for model inference."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class InferenceConfig:
    """Configuration for inference."""
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    max_batch_size: int = 1
    enable_xformers: bool = True
    enable_tiling: bool = False  # For memory optimization

@dataclass
class InferenceResult:
    """Result from inference."""
    success: bool
    output: Optional[np.ndarray] = None  # Video frames
    metadata: Dict[str, Any] = None
    error: Optional[str] = None

class BaseInference(ABC):
    """Base class for all inference engines."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self._loaded = False
    
    @abstractmethod
    async def load_model(self, model_path: Path) -> None:
        """Load the model weights."""
        pass
    
    @abstractmethod
    async def unload_model(self) -> None:
        """Unload model to free memory."""
        pass
    
    @abstractmethod
    async def infer(
        self,
        **kwargs
    ) -> InferenceResult:
        """Run inference."""
        pass
    
    def get_device(self) -> torch.device:
        """Get the device for inference."""
        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif self.config.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    async def ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
```

##### 4.4.2 Model Loader Utility

Create `backend/services/inference/model_loader.py`:

```python
"""Utilities for loading and managing models."""
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import gc

class ModelRegistry:
    """Registry of loaded models with memory management."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models: Dict[str, Any] = {}
            cls._instance._memory_threshold = 0.9  # 90% usage triggers cleanup
        return cls._instance
    
    def register(self, name: str, model: Any) -> None:
        """Register a loaded model."""
        self._models[name] = model
        logger.info(f"Registered model: {name}")
    
    def get(self, name: str) -> Optional[Any]:
        """Get a registered model."""
        return self._models.get(name)
    
    def unload(self, name: str) -> bool:
        """Unload a model to free memory."""
        if name in self._models:
            del self._models[name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            logger.info(f"Unloaded model: {name}")
            return True
        return False
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        usage = {}
        if torch.cuda.is_available():
            usage["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            usage["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            usage["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        return usage

def load_checkpoint(
    path: Path,
    device: torch.device,
    dtype: torch.dtype = torch.float16
) -> Dict[str, torch.Tensor]:
    """Load a model checkpoint."""
    logger.info(f"Loading checkpoint: {path}")
    
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(path), device=str(device))
    else:
        state_dict = torch.load(path, map_location=device)
    
    # Convert to target dtype
    for key in state_dict:
        if state_dict[key].dtype == torch.float32:
            state_dict[key] = state_dict[key].to(dtype)
    
    return state_dict
```

#### Phase 2: Implement Wan VACE Inference (Weeks 2-3)

##### 4.4.3 Wan VACE Direct Inference

Create `backend/services/inference/wan_vace.py`:

```python
"""Direct inference for Wan VACE models."""
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Callable
from PIL import Image

from .base import BaseInference, InferenceConfig, InferenceResult
from .model_loader import load_checkpoint, ModelRegistry

class WanVACEInference(BaseInference):
    """Direct inference for Wan 2.1 VACE pose/motion transfer."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.vae = None
        self.text_encoder = None
        self.unet = None
        self.image_encoder = None
    
    async def load_model(self, model_path: Path) -> None:
        """Load Wan VACE model components."""
        from diffusers import AutoencoderKL
        from transformers import CLIPTextModel, CLIPTokenizer
        
        device = self.get_device()
        dtype = self.config.dtype
        
        # Load VAE
        vae_path = model_path / "vae"
        self.vae = AutoencoderKL.from_pretrained(
            vae_path, 
            torch_dtype=dtype
        ).to(device)
        
        # Load text encoder
        text_encoder_path = model_path / "text_encoder"
        self.text_encoder = CLIPTextModel.from_pretrained(
            text_encoder_path,
            torch_dtype=dtype
        ).to(device)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder_path)
        
        # Load UNet (video diffusion model)
        unet_path = model_path / "unet"
        # This would be Wan's specific video UNet
        self.unet = self._load_wan_unet(unet_path, device, dtype)
        
        # Load image encoder for reference conditioning
        image_encoder_path = model_path / "image_encoder"
        self.image_encoder = self._load_image_encoder(image_encoder_path, device, dtype)
        
        self._loaded = True
    
    def _load_wan_unet(self, path: Path, device: torch.device, dtype: torch.dtype):
        """Load Wan's video UNet architecture."""
        # This requires Wan's specific UNet implementation
        # Would need to port from their official repo
        from wan.models import VideoUNet3D
        
        model = VideoUNet3D.from_pretrained(path, torch_dtype=dtype)
        model = model.to(device)
        
        if self.config.enable_xformers:
            model.enable_xformers_memory_efficient_attention()
        
        return model
    
    def _load_image_encoder(self, path: Path, device: torch.device, dtype: torch.dtype):
        """Load image encoder for reference."""
        from transformers import CLIPVisionModel
        
        return CLIPVisionModel.from_pretrained(
            path, torch_dtype=dtype
        ).to(device)
    
    async def encode_reference(self, image: np.ndarray) -> torch.Tensor:
        """Encode reference image for conditioning."""
        await self.ensure_loaded()
        
        # Preprocess image
        image = Image.fromarray(image)
        # ... preprocessing
        
        with torch.no_grad():
            embedding = self.image_encoder(image_tensor).last_hidden_state
        
        return embedding
    
    async def infer(
        self,
        reference_image: np.ndarray,
        control_frames: List[np.ndarray],  # Pose frames
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 48,
        steps: int = 25,
        cfg_scale: float = 7.0,
        strength: float = 0.85,
        seed: int = -1,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> InferenceResult:
        """Run video generation inference."""
        await self.ensure_loaded()
        
        device = self.get_device()
        
        try:
            # Encode reference image
            if progress_callback:
                progress_callback(5, "Encoding reference image")
            
            ref_embedding = await self.encode_reference(reference_image)
            
            # Encode text prompt
            if progress_callback:
                progress_callback(10, "Encoding prompts")
            
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            text_embedding = self.text_encoder(text_inputs.input_ids)[0]
            
            # Encode control frames (poses)
            if progress_callback:
                progress_callback(15, "Processing control frames")
            
            control_latents = await self._encode_control_frames(control_frames)
            
            # Initialize noise
            if seed >= 0:
                torch.manual_seed(seed)
            
            latent_shape = (1, 4, num_frames, 64, 64)
            latents = torch.randn(latent_shape, device=device, dtype=self.config.dtype)
            
            # Diffusion loop
            from diffusers import DDIMScheduler
            scheduler = DDIMScheduler.from_pretrained(
                self.config.model_path / "scheduler"
            )
            scheduler.set_timesteps(steps)
            
            for i, t in enumerate(scheduler.timesteps):
                if progress_callback:
                    pct = 20 + (i / len(scheduler.timesteps)) * 70
                    progress_callback(pct, f"Denoising step {i+1}/{steps}")
                
                # Predict noise
                with torch.no_grad():
                    noise_pred = self.unet(
                        latents,
                        t,
                        encoder_hidden_states=text_embedding,
                        image_embeds=ref_embedding,
                        control_latents=control_latents,
                    ).sample
                
                # Classifier-free guidance
                # ... (implement CFG)
                
                # Step
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            # Decode latents to video
            if progress_callback:
                progress_callback(90, "Decoding video")
            
            with torch.no_grad():
                video_frames = self.vae.decode(latents).sample
            
            # Post-process
            video_frames = self._postprocess_video(video_frames)
            
            if progress_callback:
                progress_callback(100, "Complete")
            
            return InferenceResult(
                success=True,
                output=video_frames,
                metadata={"num_frames": num_frames, "steps": steps}
            )
            
        except Exception as e:
            return InferenceResult(
                success=False,
                error=str(e)
            )
    
    async def _encode_control_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Encode control (pose) frames."""
        # Process pose frames for conditioning
        # ...
        pass
    
    def _postprocess_video(self, latents: torch.Tensor) -> np.ndarray:
        """Convert latents to video frames."""
        # Scale and convert to numpy
        frames = ((latents + 1) / 2 * 255).clamp(0, 255)
        frames = frames.permute(0, 2, 3, 4, 1).cpu().numpy().astype(np.uint8)
        return frames[0]  # Remove batch dim
    
    async def unload_model(self) -> None:
        """Unload model to free memory."""
        self.vae = None
        self.text_encoder = None
        self.unet = None
        self.image_encoder = None
        self._loaded = False
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
```

#### Phase 3: Implement LivePortrait Direct Inference (Week 3)

##### 4.4.4 LivePortrait Direct Inference

Create `backend/services/inference/liveportrait.py`:

```python
"""Direct inference for LivePortrait."""
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import cv2

from .base import BaseInference, InferenceConfig, InferenceResult

class LivePortraitInference(BaseInference):
    """Direct inference for LivePortrait face animation."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.appearance_extractor = None
        self.motion_extractor = None
        self.warping_module = None
        self.spade_generator = None
        self.stitching_module = None
    
    async def load_model(self, model_path: Path) -> None:
        """Load LivePortrait model components."""
        device = self.get_device()
        dtype = self.config.dtype
        
        # Import LivePortrait modules
        # (Would need to port from official repo or use their package)
        from liveportrait.modules import (
            AppearanceFeatureExtractor,
            MotionExtractor,
            WarpingModule,
            SPADEGenerator,
            StitchingModule
        )
        
        self.appearance_extractor = AppearanceFeatureExtractor()
        self.appearance_extractor.load_state_dict(
            torch.load(model_path / "appearance.pth", map_location=device)
        )
        self.appearance_extractor.to(device, dtype)
        
        self.motion_extractor = MotionExtractor()
        self.motion_extractor.load_state_dict(
            torch.load(model_path / "motion.pth", map_location=device)
        )
        self.motion_extractor.to(device, dtype)
        
        self.warping_module = WarpingModule()
        self.warping_module.load_state_dict(
            torch.load(model_path / "warping.pth", map_location=device)
        )
        self.warping_module.to(device, dtype)
        
        self.spade_generator = SPADEGenerator()
        self.spade_generator.load_state_dict(
            torch.load(model_path / "generator.pth", map_location=device)
        )
        self.spade_generator.to(device, dtype)
        
        self.stitching_module = StitchingModule()
        self.stitching_module.load_state_dict(
            torch.load(model_path / "stitching.pth", map_location=device)
        )
        self.stitching_module.to(device, dtype)
        
        self._loaded = True
    
    async def prepare_source(self, source_image: np.ndarray) -> dict:
        """
        Prepare source image for animation.
        
        This extracts the appearance features that will be animated.
        """
        await self.ensure_loaded()
        device = self.get_device()
        
        # Preprocess
        source_tensor = self._preprocess(source_image).to(device)
        
        with torch.no_grad():
            # Extract appearance features
            appearance = self.appearance_extractor(source_tensor)
            
            # Extract canonical motion (neutral expression)
            source_motion = self.motion_extractor(source_tensor)
        
        return {
            "appearance": appearance,
            "source_motion": source_motion,
            "source_image": source_tensor,
        }
    
    async def animate_frame(
        self,
        prepared_source: dict,
        driving_frame: np.ndarray,
        relative_motion: bool = True,
    ) -> np.ndarray:
        """
        Animate source with driving frame motion.
        
        Args:
            prepared_source: Output from prepare_source()
            driving_frame: Frame containing expressions to transfer
            relative_motion: Use relative motion (recommended)
        
        Returns:
            Animated frame as numpy array
        """
        await self.ensure_loaded()
        device = self.get_device()
        
        # Preprocess driving frame
        driving_tensor = self._preprocess(driving_frame).to(device)
        
        with torch.no_grad():
            # Extract driving motion
            driving_motion = self.motion_extractor(driving_tensor)
            
            if relative_motion:
                # Relative motion: driving - source + source
                motion = driving_motion - prepared_source["source_motion"] + prepared_source["source_motion"]
            else:
                motion = driving_motion
            
            # Warp features
            warped_features = self.warping_module(
                prepared_source["appearance"],
                motion
            )
            
            # Generate output
            output = self.spade_generator(warped_features)
            
            # Stitch onto original (preserve non-face regions)
            final = self.stitching_module(
                output,
                prepared_source["source_image"]
            )
        
        return self._postprocess(final)
    
    async def infer(
        self,
        source_image: np.ndarray,
        driving_frames: list,
        **kwargs
    ) -> InferenceResult:
        """Run full video animation."""
        try:
            prepared = await self.prepare_source(source_image)
            
            output_frames = []
            for i, frame in enumerate(driving_frames):
                animated = await self.animate_frame(prepared, frame)
                output_frames.append(animated)
            
            return InferenceResult(
                success=True,
                output=np.stack(output_frames),
                metadata={"num_frames": len(output_frames)}
            )
        except Exception as e:
            return InferenceResult(success=False, error=str(e))
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize to 256x256
        image = cv2.resize(image, (256, 256))
        # Normalize to [-1, 1]
        tensor = torch.from_numpy(image).float() / 127.5 - 1.0
        # HWC -> BCHW
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor
    
    def _postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """Postprocess model output to image."""
        # BCHW -> HWC
        image = tensor.squeeze(0).permute(1, 2, 0)
        # [-1, 1] -> [0, 255]
        image = ((image + 1) * 127.5).clamp(0, 255)
        return image.cpu().numpy().astype(np.uint8)
    
    async def unload_model(self) -> None:
        """Unload model."""
        self.appearance_extractor = None
        self.motion_extractor = None
        self.warping_module = None
        self.spade_generator = None
        self.stitching_module = None
        self._loaded = False
```

#### Phase 4: Update Video Generation Tasks (Week 4)

##### 4.4.5 Update Task Router

Create `backend/workers/inference_tasks.py`:

```python
"""Video generation tasks using direct inference (no ComfyUI)."""
import asyncio
from pathlib import Path
from typing import Dict, Any
from loguru import logger

from backend.core.config import settings
from backend.core.models import JobStatus, GenerationMode, QualityPreset
from backend.services.inference.base import InferenceConfig, InferenceResult
from backend.services.inference.wan_vace import WanVACEInference
from backend.services.inference.liveportrait import LivePortraitInference
# from backend.services.inference.face_swap import FaceSwapInference

# Model singletons
_wan_vace: WanVACEInference = None
_liveportrait: LivePortraitInference = None

async def get_inference_engine(mode: GenerationMode):
    """Get the appropriate inference engine."""
    global _wan_vace, _liveportrait
    
    config = InferenceConfig(
        device=settings.device,
        dtype=torch.float16 if settings.enable_fp16 else torch.float32,
        enable_xformers=settings.enable_xformers,
    )
    
    if mode in [GenerationMode.VACE_POSE_TRANSFER, GenerationMode.VACE_MOTION_TRANSFER]:
        if _wan_vace is None:
            _wan_vace = WanVACEInference(config)
            await _wan_vace.load_model(Path(settings.wan_vace_model_path))
        return _wan_vace
    
    elif mode == GenerationMode.LIVEPORTRAIT:
        if _liveportrait is None:
            _liveportrait = LivePortraitInference(config)
            await _liveportrait.load_model(Path(settings.liveportrait_model_path))
        return _liveportrait
    
    # ... other modes
    
    raise ValueError(f"No inference engine for mode: {mode}")

async def process_video_generation_direct(job_id: str) -> None:
    """
    Video generation without ComfyUI.
    
    Uses direct Python inference instead.
    """
    from backend.services.job_manager import get_job_manager
    from backend.services.file_manager import get_file_manager
    from backend.services.video_processor import get_video_processor
    from backend.services.pose_extractor import get_pose_extractor
    
    job_manager = get_job_manager()
    file_manager = get_file_manager()
    video_processor = get_video_processor()
    pose_extractor = get_pose_extractor()
    
    try:
        job = job_manager.get_job(job_id)
        request = job.request
        
        await job_manager.update_status(job_id, JobStatus.PROCESSING, 0, "Starting")
        
        # Load inputs
        ref_file = file_manager.get_file(request.reference_image_id)
        ref_image = load_image(ref_file.path)
        
        input_frames = []
        if request.input_video_id:
            video_file = file_manager.get_file(request.input_video_id)
            async for idx, frame in video_processor.extract_frames_as_arrays(
                video_file.path,
                fps=request.fps,
                max_frames=int((request.duration or 5) * request.fps)
            ):
                input_frames.append(frame)
        
        # Get inference engine
        engine = await get_inference_engine(request.mode)
        
        # Define progress callback
        async def on_progress(pct: float, step: str):
            await job_manager.update_progress(job_id, pct, step)
        
        # Run inference
        if request.mode in [GenerationMode.VACE_POSE_TRANSFER, GenerationMode.VACE_MOTION_TRANSFER]:
            # Extract poses first
            await job_manager.update_progress(job_id, 5, "Extracting poses")
            pose_sequence = await pose_extractor.extract_from_video(
                video_file.path, fps=request.fps
            )
            control_frames = pose_extractor.poses_to_controlnet_format(pose_sequence)
            
            result = await engine.infer(
                reference_image=ref_image,
                control_frames=control_frames,
                prompt=request.prompt,
                num_frames=len(control_frames),
                steps=get_quality_settings(request.quality)["steps"],
                cfg_scale=get_quality_settings(request.quality)["cfg_scale"],
                strength=request.strength,
                seed=request.seed or -1,
                progress_callback=on_progress,
            )
        
        elif request.mode == GenerationMode.LIVEPORTRAIT:
            result = await engine.infer(
                source_image=ref_image,
                driving_frames=input_frames,
            )
        
        if not result.success:
            raise Exception(result.error)
        
        # Save output
        output_dir = file_manager.get_output_path(job_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"output.{request.output_format.value}"
        await video_processor.arrays_to_video(
            list(result.output),
            output_path,
            fps=request.fps
        )
        
        # Generate thumbnail
        thumb_path = output_dir / "thumb.jpg"
        await video_processor.generate_thumbnail(output_path, thumb_path)
        
        # Complete job
        await job_manager.complete_job(
            job_id,
            result_url=f"/outputs/{job_id}/{output_path.name}",
            thumbnail_url=f"/outputs/{job_id}/{thumb_path.name}",
        )
        
    except Exception as e:
        logger.exception(f"Generation failed: {e}")
        await job_manager.fail_job(job_id, str(e))
```

### 4.5 Configuration Updates

Update `backend/core/config.py`:

```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    # Inference backend
    use_comfyui: bool = False  # Toggle between ComfyUI and direct inference
    
    # Direct inference model paths
    wan_vace_model_path: str = "models/wan2.1_vace"
    wan_r2v_model_path: str = "models/wan2.6_r2v"
    liveportrait_model_path: str = "models/liveportrait"
    face_swap_model_path: str = "models/inswapper"
    
    # Memory management
    auto_unload_models: bool = True
    model_unload_timeout: int = 300  # seconds
```

### 4.6 Migration Strategy

1. **Phase 1**: Keep ComfyUI as default, add `use_comfyui=False` option
2. **Phase 2**: Implement and test direct inference alongside ComfyUI
3. **Phase 3**: Switch default to direct inference
4. **Phase 4**: Deprecate ComfyUI support (keep for backwards compatibility)

### 4.7 Benefits of Direct Inference

| Aspect | ComfyUI | Direct Inference |
|--------|---------|------------------|
| **Latency** | Network overhead | Minimal |
| **Memory** | Full ComfyUI process | Only required models |
| **Debugging** | Harder to debug | Full Python stack traces |
| **Dependencies** | ComfyUI + custom nodes | Python packages only |
| **Real-time** | Limited | Better suited |
| **Deployment** | Docker complexity | Simpler |
| **Flexibility** | Node-based | Full code control |

### 4.8 Challenges & Mitigations

| Challenge | Mitigation |
|-----------|------------|
| Model implementation complexity | Port from official repos, use existing packages |
| Memory management | Implement model unloading, use mixed precision |
| API compatibility | Maintain same task interface |
| Testing | Comprehensive integration tests |
| Model updates | Create model versioning system |

---

## 5. Implementation Timeline

### Week 1-2: Critical Fixes + Phase 1 Infrastructure
- Fix security issues
- Rename shadowed exceptions
- Create base inference classes
- Add health endpoints

### Week 3-4: Implement Direct Inference
- Implement Wan VACE direct inference
- Implement LivePortrait direct inference
- Create model loading utilities

### Week 5-6: MacOS Optimization
- Add MPS device detection
- Update dependencies for ARM64
- Create macOS setup scripts
- Test on Apple Silicon

### Week 7-8: Integration & Testing
- Integrate direct inference with existing API
- Comprehensive testing
- Performance benchmarking
- Documentation updates

### Week 9-10: Production Hardening
- Add database persistence
- Implement rate limiting
- Add authentication
- Final testing & deployment

---

## 6. Monitoring Checklist

After implementation, verify:

- [ ] All placeholder code replaced with working implementations
- [ ] All identified bugs fixed
- [ ] macOS Apple Silicon tested on M1/M2/M3
- [ ] Direct inference works without ComfyUI
- [ ] Memory usage is within acceptable limits
- [ ] Real-time mode achieves target FPS
- [ ] All tests pass
- [ ] Documentation updated
