# Codeplace & Optimization Plan (Revised)

## 1. Issues & Bugs Identified

### Placeholders & Incomplete Logic
- **`backend/api/websocket.py`**: The real-time processing logic (`process_frame`, `initialize_realtime_processor`) is largely stubbed out. The model initialization code is commented out.
- **`backend/services/job_manager.py`**: Contains `pass` blocks in error handling which might swallow important errors.
- **`backend/api/main.py`**: Shutdown event contains a `pass` placeholder.

### Dependencies
- **CUDA/NVIDIA Hardcoding**: The codebase assumes an NVIDIA GPU environment (`cuda`, `nvidia-smi`, `onnxruntime-gpu`).
- **ComfyUI Dependency**: The entire generation pipeline is tightly coupled to an external ComfyUI instance, adding significant overhead and complexity.

## 2. MacOS Apple Silicon (MPS) Optimization Plan

The goal is to enable the application to run efficiently on macOS devices with Apple Silicon (M1/M2/M3) using Metal Performance Shaders (MPS).

### 2.1 Configuration Changes
- **Device Selection**: Update `backend/core/config.py` to allow `mps` as a valid device type.
- **Auto-Detection**: Update `backend/services/face_detector.py` and `pose_extractor.py` to automatically detect `mps` availability using `torch.backends.mps.is_available()`.

### 2.2 Dependency Updates
- **ONNX Runtime**: Replace `onnxruntime-gpu` with `onnxruntime` (which supports CoreML/CPU) or `onnxruntime-silicon` in `backend/requirements.txt`.
- **Torch**: Ensure `torch` and `torchvision` are installed without CUDA dependencies for Mac users (usually standard `pip install torch` works, but specific versions might be needed).

### 2.3 Docker & Environment
- **Docker Compose**: Create a `docker-compose.mac.yml` or modify the existing one to be conditional. Remove `deploy: resources: reservations: devices: - driver: nvidia` sections for Mac users.
- **Setup Script**: Update `scripts/setup.sh` to check for `sw_vers` (macOS) and skip `nvidia-smi` checks.

## 3. ComfyUI Removal Plan

The goal is to remove the ComfyUI dependency and run inference directly within the Python backend using native libraries.

### 3.1 Architecture Shift
- **Current**: Backend -> ComfyUI Client -> ComfyUI Server -> Nodes -> Models.
- **Target**: Backend -> Inference Services -> Models (loaded in memory/VRAM).

### 3.2 Replacement Strategy by Feature

#### A. Face Swap (Deep Live Cam)
- **Current**: ComfyUI `ReActorFaceSwap` node.
- **Replacement**: Use `insightface` library directly.
- **Implementation**:
    1.  Load `inswapper_128.onnx` using `insightface.model_zoo`.
    2.  Use `GFPGAN` or `CodeFormer` Python packages for face restoration.
    3.  Create `backend/services/inference/face_swapper.py`.

#### B. Live Portrait
- **Current**: ComfyUI `LivePortrait` custom nodes.
- **Replacement**: Use `LivePortrait` Python codebase.
- **Implementation**:
    1.  Clone/Import `LivePortrait` core logic.
    2.  Load weights (`appearance_feature_extractor`, `motion_extractor`, etc.) directly via Torch.
    3.  Create `backend/services/inference/live_portrait.py`.

#### C. Video Generation (Wan R2V)
- **Current**: ComfyUI `WanR2V` nodes.
- **Replacement**: Use `diffusers` (HuggingFace) or native Wan implementation.
- **Implementation**:
    1.  Check `diffusers` support for Wan2.1. If not available, adapt the official Wan inference code.
    2.  Implement efficient loading (offloading to CPU when not in use) to manage memory on Mac (Unified Memory helps here).
    3.  Create `backend/services/inference/video_generator.py`.

## 4. Detailed Actionable Steps

### Phase 1: Preparation & Cleanup
1.  **Rectify Placeholders**:
    -   Add proper error logging to `backend/api/websocket.py`.
    -   Implement graceful shutdown in `backend/api/main.py`.
2.  **Environment Adaptation**:
    -   Modify `backend/core/config.py` to support `mps`.
    -   Update `backend/requirements.txt`: Remove `onnxruntime-gpu`, add `onnxruntime`, `insightface`, `gfpgan`, `diffusers`, `transformers`.

### Phase 2: Direct Inference Implementation
3.  **Model Manager**:
    -   Create `backend/services/model_manager.py` to handle loading/unloading models to manage VRAM/Unified Memory.
4.  **Implement Face Swap**:
    -   Port the logic from `ReActor` to a native Python class using `insightface`.
5.  **Implement Live Portrait**:
    -   Download LivePortrait weights to `models/liveportrait`.
    -   Wrap the inference loop in a Python service.
6.  **Implement Video Gen**:
    -   Set up the `Wan` pipeline using `diffusers`.

### Phase 3: Integration & ComfyUI Removal
7.  **Update Workers**:
    -   Rewrite `backend/workers/video_tasks.py` to call the new inference services instead of `ComfyUIClient`.
8.  **Update Realtime API**:
    -   Update `backend/api/routes/realtime.py` and `websocket.py` to use the new `FaceSwap` and `LivePortrait` services for low-latency processing.
9.  **Decommission ComfyUI**:
    -   Remove `comfyui/` directory.
    -   Remove `backend/services/comfyui_client.py`.
    -   Remove `comfyui` service from `docker-compose.yml`.

### Phase 4: Optimization
10. **Mac Optimization**:
    -   Ensure `torch.set_default_device('mps')` is used where appropriate.
    -   Use `fp16` (half precision) on MPS for performance.
