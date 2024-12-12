from typing import List, Dict, Any, Optional
import json
import asyncio
from .globals import get_model_memory_manager

class ModelCommandHandler:
    def __init__(self):
        self.model_manager = get_model_memory_manager()
        
    async def handle_command(self, command_data: bytes) -> Optional[bytes]:
        try:
            print(f"Command data: {command_data}")
            command = command_data
            
            if command["command"] == "load":
                await self._handle_load(command["model_ids"], command["priority"])
                return {
                    "status": "done",
                    "message": "Models loaded"
                }
                
            elif command["command"] == "unload":
                await self._handle_unload(command["model_ids"])
                return {
                    "status": "done",
                    "message": "Models unloaded"
                }
                
            elif command["command"] == "enable":
                await self._handle_enable(command["model_ids"])
                return {
                    "status": "done",
                    "message": "Models enabled"
                }
                
            elif command["command"] == "status":
                return await self._handle_status(command["model_ids"])
            
            else:
                raise ValueError(f"Unknown command: {command['command']}")
                
        except Exception as e:
            raise Exception(f"Failed to handle command: {str(e)}")

    async def _handle_load(self, model_ids: List[str], priority: bool):
        """Handle loading models with priority flag"""
        for model_id in model_ids:
            await self.model_manager.load(model_id)
            
    async def _handle_unload(self, model_ids: List[str]):
        """Handle unloading models"""
        for model_id in model_ids:
            print(f"Unloading model: {model_id}")
            self.model_manager.unload(model_id)
            
    async def _handle_enable(self, model_ids: List[str]):
        """Handle enabling models"""
        for model_id in model_ids:
            # TODO: this is not the right function
            await self.model_manager.warmup_pipeline(model_id)
            
    async def _handle_status(self, model_ids: List[str]) -> bytes:
        """Handle getting model status"""
        loaded_models = []
                
        # GPU models
        for model_id in self.model_manager.loaded_models:
            memory_gb = self._get_model_memory_usage(model_id)
            loaded_models.append({
                "model_id": model_id,
                "location": "gpu",
                "memory_usage": round(memory_gb, 2)
            })

        # CPU models
        for model_id in self.model_manager.cpu_models:
            if model_id not in self.model_manager.loaded_models:
                memory_gb = self._get_model_memory_usage(model_id)
                loaded_models.append({
                    "model_id": model_id,
                    "location": "cpu",
                    "memory_usage": round(memory_gb, 2)
                })

        return loaded_models
        
    def _get_model_location(self, model_id: str) -> str:
        if model_id in self.model_manager.loaded_models:
            return "gpu"
        elif model_id in self.model_manager.cpu_models:
            return "cpu"
        return "unloaded"
        
    def _get_model_memory_usage(self, model_id: str) -> int:
        return self.model_manager.model_sizes.get(model_id, 0)