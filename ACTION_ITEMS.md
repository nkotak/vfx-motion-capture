# Action Items - Prioritized Task List

## 🔴 Critical Priority (P0) - Start Immediately

### 1. Fix Critical Bugs
- [ ] **BUG-001:** Add Apple Silicon (MPS) device detection
  - File: `backend/core/config.py`
  - File: `backend/api/routes/realtime.py`
  - Time: 4 hours
  - Dependencies: None

- [ ] **BUG-002:** Verify/fix frame decoding typo (if exists)
  - File: `backend/api/websocket.py:336`
  - Time: 15 minutes
  - Dependencies: None

### 2. Implement Placeholder Code
- [ ] **PLACEHOLDER-001:** Implement LivePortrait processor
  - File: `backend/api/websocket.py` (lines 297-302)
  - New File: `backend/services/liveportrait_processor.py`
  - Time: 2-3 days
  - Dependencies: LivePortrait model files

- [ ] **PLACEHOLDER-002:** Complete face swap implementation
  - File: `backend/api/websocket.py` (lines 369-377)
  - File: `backend/services/face_detector.py`
  - Time: 2-3 days
  - Dependencies: Face alignment library

## 🟠 High Priority (P1) - Fix This Week

### 3. Error Handling & Stability
- [ ] **BUG-003:** Add error handling to ComfyUI client downloads
  - File: `backend/services/comfyui_client.py`
  - Time: 4 hours
  - Dependencies: None

- [ ] **BUG-004:** Fix race condition in job manager
  - File: `backend/services/job_manager.py`
  - Time: 2 hours
  - Dependencies: None

- [ ] **BUG-006:** Fix memory leaks in WebSocket handlers
  - File: `backend/api/websocket.py`
  - Time: 1 day
  - Dependencies: None

- [ ] **BUG-007:** Add input validation for generation requests
  - File: `backend/api/routes/generate.py`
  - Time: 4 hours
  - Dependencies: None

## 🟡 Medium Priority (P2) - Fix This Month

### 4. Code Quality
- [ ] **BUG-005:** Fix video processor frame pattern issue
  - File: `backend/services/video_processor.py`
  - Time: 2 hours
  - Dependencies: None

- [ ] Add comprehensive type hints
  - Files: All Python files
  - Time: 2 days
  - Dependencies: None

- [ ] Add unit tests for critical paths
  - New Directory: `backend/tests/`
  - Time: 3 days
  - Dependencies: pytest

- [ ] Add rate limiting to API endpoints
  - File: `backend/api/main.py`
  - Time: 4 hours
  - Dependencies: slowapi or similar

## 🟢 Apple Silicon Support - 4 Weeks

### Week 1: Device Detection & Configuration
- [ ] Update configuration system for MPS
  - File: `backend/core/config.py`
  - Time: 4 hours

- [ ] Update GPU detection endpoint
  - File: `backend/api/routes/realtime.py`
  - Time: 2 hours

### Week 2: Model & Framework Updates
- [ ] Update PyTorch dependencies
  - File: `backend/requirements.txt`
  - Time: 1 day

- [ ] Update face detector for CoreML provider
  - File: `backend/services/face_detector.py`
  - Time: 1 day

- [ ] Update pose extractor for MPS
  - File: `backend/services/pose_extractor.py`
  - Time: 1 day

### Week 3: Performance Optimization
- [ ] Optimize for unified memory
  - Files: Multiple
  - Time: 2 days

- [ ] Update Docker configuration
  - File: `docker-compose.yml`
  - Time: 1 day

- [ ] Update setup scripts
  - File: `scripts/setup.sh`
  - Time: 4 hours

### Week 4: Testing & Validation
- [ ] Create test suite for Apple Silicon
  - Time: 2 days

- [ ] Update documentation
  - Files: `README.md`, `PLAN.md`
  - Time: 1 day

## 🔵 ComfyUI Removal (Optional) - 5-6 Weeks

### Week 1-2: Model Integration Layer
- [ ] Create base model interface
  - New File: `backend/services/models/base_model.py`
  - Time: 4 hours

- [ ] Implement Wan VACE model wrapper
  - New File: `backend/services/models/wan_vace.py`
  - Time: 3-4 days

- [ ] Implement LivePortrait model wrapper
  - New File: `backend/services/models/liveportrait.py`
  - Time: 2-3 days

- [ ] Implement Wan R2V model wrapper
  - New File: `backend/services/models/wan_r2v.py`
  - Time: 3-4 days

### Week 3: Workflow Orchestration
- [ ] Create workflow engine
  - New File: `backend/services/workflow_engine.py`
  - Time: 3-4 days

- [ ] Update video tasks
  - File: `backend/workers/video_tasks.py`
  - Time: 2 days

### Week 4: Remove Dependencies
- [ ] Remove ComfyUI client
  - File: `backend/services/comfyui_client.py`
  - Time: 1 hour

- [ ] Update configuration
  - File: `backend/core/config.py`
  - Time: 2 hours

- [ ] Update Docker configuration
  - File: `docker-compose.yml`
  - Time: 1 hour

- [ ] Update documentation
  - Files: `README.md`, `PLAN.md`
  - Time: 4 hours

### Week 5: Model Management
- [ ] Create model manager
  - New File: `backend/services/model_manager.py`
  - Time: 2 days

## 📋 Testing Checklist

- [ ] Unit tests for device detection
- [ ] Unit tests for model loading
- [ ] Integration tests for workflows
- [ ] Performance benchmarks
- [ ] Compatibility tests (macOS, Linux, Windows)
- [ ] Apple Silicon specific tests
- [ ] Memory leak tests
- [ ] Error handling tests

## 📚 Documentation Updates

- [ ] Update README.md with Apple Silicon instructions
- [ ] Update PLAN.md architecture diagrams
- [ ] Add troubleshooting guide
- [ ] Update API documentation
- [ ] Add model setup guide
- [ ] Create developer onboarding guide

## 🎯 Success Criteria

### Bug Fixes
- ✅ Zero critical bugs
- ✅ <5 medium priority bugs
- ✅ 90%+ test coverage

### Apple Silicon Support
- ✅ All features work on Apple Silicon
- ✅ Performance within 20% of CUDA equivalent
- ✅ No CUDA-specific code paths

### ComfyUI Removal (if pursued)
- ✅ All workflows functional
- ✅ No performance regression
- ✅ Simplified deployment

---

## Notes

- All time estimates are for a single developer
- Dependencies should be resolved before starting tasks
- Test after each major change
- Update documentation as you go
- Use feature branches for each major task

## Quick Start

1. Start with P0 tasks (Critical Priority)
2. Test on target hardware (Apple Silicon if applicable)
3. Move to P1 tasks after P0 complete
4. Consider ComfyUI removal based on project needs

---

For detailed implementation plans, see `ANALYSIS_AND_PLANS.md`.
For quick reference, see `QUICK_REFERENCE_CHECKLIST.md`.
