"""
Wan Video Generation Service.
Uses Diffusers (if supported) or native implementation.
"""

import torch
import numpy as np
from typing import Optional
from loguru import logger
from backend.core.config import settings
from backend.services.device import resolve_device
from backend.services.model_manager import get_model_manager

class WanVideoService:
    def __init__(self):
        self.model_manager = get_model_manager()
        self.device = resolve_device(settings.device)
        
    def _load_pipeline(self):
        def loader():
            try:
                from diffusers import DiffusionPipeline
                # Placeholder for Wan support in diffusers
                # If Wan is not yet in official diffusers, this would fail
                model_id = "Wan-AI/Wan2.1-T2V-14B"
                pipe = DiffusionPipeline.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16 if settings.enable_fp16 else torch.float32
                )
                pipe.to(self.device)
                return pipe
            except ImportError:
                logger.warning("Diffusers not installed.")
                return None
            except Exception as e:
                logger.warning(f"Failed to load Wan pipeline: {e}")
                return None
                
        return self.model_manager.load_model("wan_video", loader)

    async def generate(self, prompt: str, reference_image: Optional[np.ndarray] = None) -> str:
        """
        Generate video from prompt and optional reference image.
        """
        pipe = self._load_pipeline()
        if not pipe:
            raise RuntimeError("Wan video model not available")
            
        logger.info(f"Generating video for: {prompt}")
        # Generation logic here...
        return "output_path.mp4"

_wan_service = None

def get_wan_video_service() -> WanVideoService:
    global _wan_service
    if _wan_service is None:
        _wan_service = WanVideoService()
    return _wan_service
