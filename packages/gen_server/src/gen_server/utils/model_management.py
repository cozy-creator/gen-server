import torch
from typing import Dict, Any, Optional
from enum import Enum
import logging

class ModelDevice(Enum):
    CPU = "cpu"
    CUDA = "cuda"

class ModelManager:
    def __init__(self, max_gpu_models: int = 2, max_cpu_models: int = 5):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.max_gpu_models = max_gpu_models
        self.max_cpu_models = max_cpu_models
        self.logger = logging.getLogger(__name__)

    def load_model(self, model_id: str, model: Any, device: ModelDevice = ModelDevice.CPU) -> None:
        if model_id in self.models:
            self.logger.warning(f"Model {model_id} already loaded. Updating existing model.")
        
        self.models[model_id] = {
            "model": model,
            "device": device,
            "last_used": 0
        }
        self._move_model(model_id, device)

    def get_model(self, model_id: str, preferred_device: ModelDevice = ModelDevice.CUDA) -> Optional[Any]:
        if model_id not in self.models:
            self.logger.error(f"Model {model_id} not found.")
            return None

        model_info = self.models[model_id]
        if model_info["device"] != preferred_device:
            self._move_model(model_id, preferred_device)

        model_info["last_used"] += 1
        return model_info["model"]

    def unload_model(self, model_id: str) -> None:
        if model_id in self.models:
            del self.models[model_id]
            torch.cuda.empty_cache()
            self.logger.info(f"Model {model_id} unloaded.")
        else:
            self.logger.warning(f"Model {model_id} not found. Cannot unload.")

    def _move_model(self, model_id: str, target_device: ModelDevice) -> None:
        model_info = self.models[model_id]
        model = model_info["model"]

        if model_info["device"] == target_device:
            return

        if target_device == ModelDevice.CUDA:
            if self._gpu_models_count() >= self.max_gpu_models:
                self._make_room_on_gpu()
            model.to("cuda")
            self.logger.info(f"Model {model_id} moved to GPU.")
        else:
            model.to("cpu")
            self.logger.info(f"Model {model_id} moved to CPU.")

        model_info["device"] = target_device

    def _gpu_models_count(self) -> int:
        return sum(1 for model in self.models.values() if model["device"] == ModelDevice.CUDA)

    def _make_room_on_gpu(self) -> None:
        gpu_models = [
            (model_id, info) for model_id, info in self.models.items() 
            if info["device"] == ModelDevice.CUDA
        ]
        gpu_models.sort(key=lambda x: x[1]["last_used"])

        while self._gpu_models_count() >= self.max_gpu_models and gpu_models:
            model_id, _ = gpu_models.pop(0)
            self._move_model(model_id, ModelDevice.CPU)

        if len(self.models) > self.max_cpu_models + self.max_gpu_models:
            cpu_models = [
                (model_id, info) for model_id, info in self.models.items() 
                if info["device"] == ModelDevice.CPU
            ]
            cpu_models.sort(key=lambda x: x[1]["last_used"])
            
            while len(self.models) > self.max_cpu_models + self.max_gpu_models and cpu_models:
                model_id, _ = cpu_models.pop(0)
                self.unload_model(model_id)

    def clear_all_models(self) -> None:
        self.models.clear()
        torch.cuda.empty_cache()
        self.logger.info("All models cleared from memory.")