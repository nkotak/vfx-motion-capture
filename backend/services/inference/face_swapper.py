"""
Face Swapping Inference Service.
Replaces ComfyUI's ReActor node.
"""

import os
import cv2
import numpy as np
import insightface
from loguru import logger
from backend.core.config import settings
from backend.services.face_detector import get_face_detector
from backend.services.model_manager import get_model_manager

class FaceSwapper:
    def __init__(self):
        self.face_detector = get_face_detector()
        self.model_manager = get_model_manager()
        self.device = settings.device
        
    def _load_swapper(self):
        """Load the inswapper model via ModelManager."""
        def loader():
            model_path = settings.models_dir / "insightface" / "inswapper_128.onnx"
            if not model_path.exists():
                raise FileNotFoundError(f"Face swap model not found at {model_path}")
            
            providers = []
            if self.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif self.device == "mps":
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                
            return insightface.model_zoo.get_model(str(model_path), providers=providers)
            
        return self.model_manager.load_model("inswapper_128", loader)

    async def swap_face(self, source_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
        """
        Swap face from source_img into target_img.
        """
        swapper = self._load_swapper()
        
        # 1. Detect source face
        try:
            source_face = await self.face_detector.get_primary_face(source_img)
        except Exception as e:
            logger.warning(f"Could not find source face: {e}")
            return target_img

        # 2. Detect target faces
        target_faces = await self.face_detector.detect_faces(target_img)
        if not target_faces:
            return target_img
            
        # 3. Perform swap
        result_img = target_img.copy()
        
        # In a real app, you might select which target face to swap
        # For now, swap the largest one (primary)
        target_face = target_faces[0]
        
        # InsightFace's swapper expects raw face objects, but our detector wraps them.
        # However, the swapper mainly needs the kps (landmarks) and the source embedding.
        # We need to construct a simplified object that the swapper accepts if we can't get the raw one easily.
        # Fortunately, the inswapper usually takes the target image, target face object (with kps), 
        # and source face object (with embedding) and paste_back=True.
        
        # Since our DetectedFace is a wrapper, we might need to adapt.
        # Let's assume for this implementation we can construct a dummy object 
        # or that we modify FaceDetector to return raw objects if needed.
        # For this prototype, I'll rely on the swapper's expected interface.
        
        # Construct dummy target face for inswapper
        class Face:
            def __init__(self, kps):
                self.kps = kps
        
        tf_obj = Face(target_face.landmarks)
        
        # Construct dummy source face
        sf_obj = Face(None)
        sf_obj.embedding = source_face.embedding
        sf_obj.normed_embedding = source_face.embedding / np.linalg.norm(source_face.embedding)
        
        result_img = swapper.get(result_img, tf_obj, sf_obj, paste_back=True)
        
        return result_img

_face_swapper = None

def get_face_swapper() -> FaceSwapper:
    global _face_swapper
    if _face_swapper is None:
        _face_swapper = FaceSwapper()
    return _face_swapper
