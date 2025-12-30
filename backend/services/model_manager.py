"""
Model Manager Service.
Handles loading, unloading, and caching of AI models to manage VRAM/memory efficiently.
"""

import gc
import torch
from typing import Dict, Any, Optional, Type
from loguru import logger
from backend.core.config import settings

class ModelManager:
    """
    Manages AI models to optimize memory usage.
    Singleton pattern to ensure models are shared.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.loaded_models: Dict[str, Any] = {}
        return cls._instance
    
    def __init__(self):
        pass

    def load_model(self, model_id: str, loader_func: callable, *args, **kwargs) -> Any:
        """
        Load a model if not already loaded.
        
        Args:
            model_id: Unique identifier for the model
            loader_func: Function to call to load the model
            *args, **kwargs: Arguments to pass to loader_func
            
        Returns:
            The loaded model
        """
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
            
        logger.info(f"Loading model: {model_id}")
        
        # Aggressively clean memory before loading a new model
        # In a real system, you might want to unload LRU models here
        self._clean_memory()
        
        try:
            model = loader_func(*args, **kwargs)
            self.loaded_models[model_id] = model
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

    def get_model(self, model_id: str) -> Optional[Any]:
        """Get a loaded model or None."""
        return self.loaded_models.get(model_id)
        
    def unload_model(self, model_id: str) -> bool:
        """Unload a specific model."""
        if model_id in self.loaded_models:
            logger.info(f"Unloading model: {model_id}")
            del self.loaded_models[model_id]
            self._clean_memory()
            return True
        return False
        
    def unload_all(self):
        """Unload all models."""
        logger.info("Unloading all models")
        self.loaded_models.clear()
        self._clean_memory()
        
    def _clean_memory(self):
        """Force garbage collection and clear CUDA/MPS cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
             # MPS empty_cache is not always exposed/effective but doesn't hurt
             try:
                 torch.mps.empty_cache()
             except:
                 pass

_model_manager = None

def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
