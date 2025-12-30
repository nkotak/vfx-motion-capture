"""
LivePortrait Inference Service.
"""

import numpy as np
from typing import Optional, Any
from loguru import logger
from backend.core.config import settings
from backend.services.device import resolve_device
from backend.services.model_manager import get_model_manager

class LivePortraitService:
    def __init__(self):
        self.model_manager = get_model_manager()
        self.device = resolve_device(settings.device)
        
    def _load_pipeline(self):
        """Load LivePortrait pipeline."""
        def loader():
            try:
                # This assumes LivePortrait is installed or in python path
                from liveportrait import LivePortraitPipeline
                from liveportrait.utils.camera import get_rotation_matrix
            except ImportError:
                logger.warning("LivePortrait library not found. Using mock implementation.")
                return MockLivePortraitPipeline()
            
            # Configure based on settings
            pipeline = LivePortraitPipeline(
                inference_cfg={
                    "device_id": 0 if self.device == "cuda" else "cpu", # MPS support in LP might need check
                    "flag_use_half_precision": settings.enable_fp16 and self.device == "cuda"
                }
            )
            return pipeline
            
        return self.model_manager.load_model("live_portrait", loader)

    def process(self, source_img: np.ndarray, driving_video_path: str) -> str:
        """
        Animate source image using driving video.
        Returns path to output video.
        """
        pipeline = self._load_pipeline()
        # Implementation details would go here (crop, extract features, animate, save)
        # For now, just a pass-through
        return driving_video_path

    def process_frame(self, source_img: np.ndarray, driving_frame: np.ndarray) -> np.ndarray:
        """
        Animate source image using a single driving frame.
        """
        pipeline = self._load_pipeline()
        # Stub logic
        return driving_frame


class MockLivePortraitPipeline:
    def __init__(self, *args, **kwargs):
        pass
        
    def execute(self, *args, **kwargs):
        logger.info("Mock LivePortrait execution")
        return None

_live_portrait = None

def get_live_portrait_service() -> LivePortraitService:
    global _live_portrait
    if _live_portrait is None:
        _live_portrait = LivePortraitService()
    return _live_portrait
