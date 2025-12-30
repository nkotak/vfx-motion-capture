# Quick Reference Checklist

## Critical Bugs to Fix (P0)

- [x] **Bug #2:** Fix typo `frombuffer` → `frombuffer` in `backend/api/websocket.py:336` ✅ FIXED
- [ ] **Bug #1:** Add Apple Silicon (MPS) device detection and support
- [ ] **Placeholder:** Implement LivePortrait processor (`backend/api/websocket.py:297-302`)
- [ ] **Placeholder:** Complete face swap implementation (`backend/api/websocket.py:369-377`)

## High Priority Fixes (P1)

- [ ] **Bug #3:** Add error handling to ComfyUI client downloads
- [ ] **Bug #4:** Fix race condition in job manager concurrent job counting
- [ ] **Bug #6:** Fix memory leaks in WebSocket handlers
- [ ] **Bug #7:** Add input validation for generation requests

## Medium Priority (P2)

- [ ] **Bug #5:** Fix video processor frame pattern issue
- [ ] Add comprehensive type hints
- [ ] Add unit tests
- [ ] Add rate limiting

## Apple Silicon Support Checklist

### Phase 1: Device Detection
- [ ] Update `backend/core/config.py` with MPS detection
- [ ] Update `backend/api/routes/realtime.py` compatibility check
- [ ] Test device detection on Apple Silicon

### Phase 2: Model Updates
- [ ] Update PyTorch dependencies (verify MPS support)
- [ ] Update `backend/services/face_detector.py` for CoreML provider
- [ ] Update `backend/services/pose_extractor.py` for MPS
- [ ] Test all models on Apple Silicon

### Phase 3: Performance
- [ ] Optimize for unified memory architecture
- [ ] Update Docker configuration for ARM64
- [ ] Update setup scripts for Apple Silicon
- [ ] Benchmark performance vs CUDA

## ComfyUI Removal Checklist (If Pursuing)

### Phase 1: Model Integration
- [ ] Create `backend/services/models/base_model.py`
- [ ] Implement `backend/services/models/wan_vace.py`
- [ ] Implement `backend/services/models/liveportrait.py`
- [ ] Implement `backend/services/models/wan_r2v.py`

### Phase 2: Workflow Engine
- [ ] Create `backend/services/workflow_engine.py`
- [ ] Update `backend/workers/video_tasks.py`
- [ ] Remove ComfyUI client dependencies

### Phase 3: Cleanup
- [ ] Remove ComfyUI from docker-compose.yml
- [ ] Update configuration files
- [ ] Update documentation
- [ ] Test all workflows

## Testing Checklist

- [ ] Unit tests for device detection
- [ ] Unit tests for model loading
- [ ] Integration tests for workflows
- [ ] Performance benchmarks
- [ ] Compatibility tests (macOS, Linux, Windows)

## Documentation Updates

- [ ] Update README.md with Apple Silicon instructions
- [ ] Update PLAN.md architecture diagrams
- [ ] Add troubleshooting guide
- [ ] Update API documentation

---

## Quick Commands

### Check for placeholders
```bash
grep -r "TODO\|FIXME\|XXX\|PLACEHOLDER\|NotImplemented\|pass\s*$" backend/ --include="*.py"
```

### Check for CUDA/NVIDIA hardcoding
```bash
grep -r "cuda\|CUDA\|nvidia\|NVIDIA" backend/ --include="*.py"
```

### Run tests (when implemented)
```bash
pytest backend/tests/
```

### Check device availability
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")
```
