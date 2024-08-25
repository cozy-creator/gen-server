# import torch
# from gen_server.base_types import CustomNode
# from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline
# from PIL import Image
# import numpy as np
# import os
# import json

# from gen_server.utils.device import get_torch_device


# class ImageRegenNode(CustomNode):
#     """Regenerates (inpaints) a masked area in an image using a Stable Diffusion model."""

#     def __init__(self):
#         super().__init__()
#         self.DEVICE = get_torch_device()

#     def __call__(
#         self,
#         image: torch.Tensor,
#         mask: torch.Tensor,
#         text_prompt: str,
#         strength: float = 0.8,
#         checkpoint_id: str = "stabilityai/stable-diffusion-2-inpainting",
#         guidance_scale: float = 7.5,
#         num_inference_steps: int = 50,
#         random_seed: int = None,
#     ) -> dict[str, Image.Image]:
#         """
#         Args:
#             image: Input image tensor (C, H, W) or PIL Image.
#             mask: Mask tensor (C, H, W) or PIL Image (where white is the area to inpaint).
#             text_prompt: Text prompt for the inpainting process.
#             strength: Strength of the inpainting effect (0.0 - 1.0).
#             checkpoint_id: ID of the checkpoint to use for inpainting (defaults to SD2-inpainting).
#             guidance_scale: CFG scale value.
#             num_inference_steps: Number of inference steps.
#             random_seed: Random seed for reproducibility (optional).
#         Returns:
#             A dictionary containing the inpainted PIL Image.
#         """
#         try:
#             # Convert to PIL if necessary
#             if isinstance(image, torch.Tensor):
#                 image = Image.fromarray(
#                     (image * 255).permute(1, 2, 0).numpy().astype(np.uint8)
#                 )
#             if isinstance(mask, torch.Tensor):
#                 mask = Image.fromarray(
#                     (mask * 255).permute(1, 2, 0).numpy().astype(np.uint8)
#                 )

#             # Error handling for data types
#             if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
#                 raise TypeError(
#                     "Image and mask must be either torch.Tensor or PIL.Image.Image"
#                 )

#             # Choose Pipeline Based on Checkpoint
#             if "xl" in checkpoint_id.lower():  # Check for XL in the checkpoint ID
#                 pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
#                     checkpoint_id,
#                     torch_dtype=torch.float16,
#                     variant="fp16",
#                     use_safetensors=True,
#                 )
#             else:
#                 pipe = StableDiffusionInpaintPipeline.from_pretrained(
#                     checkpoint_id,
#                 )

#             pipe = pipe.to(self.DEVICE)

#             # Run the inpainting pipeline
#             generator = (
#                 torch.Generator(device=self.DEVICE).manual_seed(random_seed)
#                 if random_seed
#                 else None
#             )
#             inpainted_image = pipe(
#                 prompt=text_prompt,
#                 image=image,
#                 mask_image=mask,
#                 strength=strength,
#                 guidance_scale=guidance_scale,
#                 num_inference_steps=num_inference_steps,
#                 generator=generator,
#             ).images[0]

#             del pipe
#             return {"inpainted_image": inpainted_image}

#         except Exception as e:
#             raise ValueError(f"Error inpainting image: {e}")

#     @staticmethod
#     def get_spec():
#         """Returns the node specification."""
#         spec_file = os.path.join(os.path.dirname(__file__), "image_regen_node.json")
#         with open(spec_file, "r", encoding="utf-8") as f:
#             spec = json.load(f)
#         return spec

from gen_server.base_types import CustomNode
from PIL import Image
import torch
import torchvision.transforms as T
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionInpaintPipeline
from gen_server.utils.model_config_manager import ModelConfigManager
from typing import Union, Dict

class ImageRegenNode(CustomNode):
    def __init__(self):
        super().__init__()
        self.config_manager = ModelConfigManager()
        self.inpaint_pipelines = {}

    def _get_inpaint_pipeline(self, model_id: str):
        if model_id not in self.inpaint_pipelines:
            model_config = self.config_manager.get_model_config(model_id)
            
            if model_config['category'] == 'sdxl':
                inpaint_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True
                )
            else:  # Assuming SD1.5 for non-SDXL models
                inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True
                )
            
            inpaint_pipeline.to("cuda")
            self.inpaint_pipelines[model_id] = inpaint_pipeline

        return self.inpaint_pipelines[model_id]

    async def __call__(self,    # type: ignore
                       image: Union[Image.Image, torch.Tensor], 
                       mask: Union[Image.Image, torch.Tensor], 
                       prompt: str, 
                       model_id: str, 
                       negative_prompt: str = "",
                       num_inference_steps: int = 30,
                       strength: float = 0.7) -> Dict[str, torch.Tensor]:
        
        pipeline = self._get_inpaint_pipeline(model_id)
        
        # Convert inputs to PIL Images if they're tensors
        if isinstance(image, torch.Tensor):
            image = T.ToPILImage()(image.squeeze(0).cpu())
        if isinstance(mask, torch.Tensor):
            mask = T.ToPILImage()(mask.squeeze(0).cpu())

        model_config = self.config_manager.get_model_config(model_id)
        
        with torch.no_grad():
            output = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                strength=strength,
                guidance_scale=model_config.get("guidance_scale", 7.5),
                output_type="pt",  # Ensure output is a PyTorch tensor
            ).images[0]
        
        # Ensure the output is a 4D tensor on CPU
        output = output.cpu()
        if output.dim() == 3:
            output = output.unsqueeze(0)
        
        return {"regenerated_image": output}
