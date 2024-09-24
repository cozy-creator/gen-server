import torch
from typing import Optional, List
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from gen_server.base_types import CustomNode
from gen_server.globals import get_model_memory_manager, get_available_torch_device
from gen_server.utils.model_config_manager import ModelConfigManager

class CogVideoXNode(CustomNode):
    def __init__(self):
        super().__init__()
        self.model_memory_manager = get_model_memory_manager()
        self.config_manager = ModelConfigManager()

    async def __call__(
        self,
        model_id: str,
        prompt: str,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 50,
        num_frames: int = 16,
        fps: int = 8,
        output_path: Optional[str] = None,
    ) -> List[torch.Tensor]:
        try:
            pipeline, should_optimize = await self.model_memory_manager.load(model_id, None)

            if pipeline is None:
                raise ValueError(f"Failed to load model {model_id}")

            if not isinstance(pipeline, CogVideoXPipeline):
                raise ValueError(f"Model {model_id} is not a CogVideoXPipeline")

            if should_optimize:
                self.model_memory_manager.apply_optimizations(pipeline)

            video_frames = pipeline(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
            ).frames[0]

            if output_path:
                export_to_video(video_frames, output_path, fps=fps)

            return video_frames

        except Exception as e:
            print(f"Error in CogVideoXNode: {str(e)}")
            raise

    # def apply_optimizations(self, pipeline: CogVideoXPipeline):
    #     device = get_available_torch_device()
    #     pipeline.enable_model_cpu_offload()
    #     pipeline.vae.enable_tiling()

    #     # Additional optimizations can be added here if needed
    #     print("Optimizations applied to CogVideoX pipeline")

