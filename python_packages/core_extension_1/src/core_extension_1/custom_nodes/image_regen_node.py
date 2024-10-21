from gen_server.base_types import CustomNode
from PIL import Image
import torch
import torchvision.transforms as T
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionInpaintPipeline
from gen_server.utils.model_config_manager import ModelConfigManager
from typing import Union, Dict
from gen_server.globals import get_model_memory_manager
from gen_server.utils.image import aspect_ratio_to_dimensions


class ImageRegenNode(CustomNode):
    def __init__(self):
        super().__init__()
        self.config_manager = ModelConfigManager()
        self.inpaint_pipelines = {}
        self.model_memory_manager = get_model_memory_manager()

        # sdxl-inpainting-1

    async def _get_inpaint_pipeline(self, model_id: str, pipe_type: str):

        inpaint_pipeline = await self.model_memory_manager.load(model_id, pipe_type=pipe_type)
        if inpaint_pipeline is None:
            raise ValueError(f"Model {model_id} not found in memory manager")

        return inpaint_pipeline

    async def __call__(self,    # type: ignore
                       image: Union[Image.Image, torch.Tensor], 
                       mask: Union[Image.Image, torch.Tensor], 
                       prompt: str, 
                       model_id: str, 
                       negative_prompt: str = "",
                       num_inference_steps: int = 25,
                       strength: float = 0.7) -> Dict[str, torch.Tensor]:
        
        pipeline = await self._get_inpaint_pipeline(model_id, pipe_type="inpaint")

        class_name = pipeline.__class__.__name__

        print(f"Using pipeline {class_name}")
        
        # Convert inputs to PIL Images if they're tensors
        if isinstance(image, torch.Tensor):
            image = T.ToPILImage()(image.squeeze(0).cpu())
        if isinstance(mask, torch.Tensor):
            mask = T.ToPILImage()(mask.squeeze(0).cpu())

        model_config = self.config_manager.get_model_config(model_id, class_name)

        self.model_memory_manager.apply_optimizations(pipeline)
        
        with torch.no_grad():
            output = pipeline(
                prompt=prompt,
                # negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                # width=image.width,
                # height=image.height,
                num_inference_steps=num_inference_steps,
                strength=strength,
                # guidance_scale=model_config.get("guidance_scale", 7.5),
                output_type="pt",  # Ensure output is a PyTorch tensor
            ).images

        self.model_memory_manager.flush_memory()
        
        # Ensure the output is a 4D tensor on CPU
        # output = output.cpu()
        # if output.dim() == 3:
        #     output = output.unsqueeze(0)
        
        return {"regenerated_image": output}
