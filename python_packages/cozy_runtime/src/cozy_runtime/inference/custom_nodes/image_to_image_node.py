from cozy_runtime.base_types import CustomNode
from PIL import Image
import torch
import torchvision.transforms as T
from typing import Union, Dict, Optional
from cozy_runtime.utils.model_config_manager import ModelConfigManager
from cozy_runtime.globals import get_model_memory_manager
from tqdm import tqdm
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    FluxImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusion3Img2ImgPipeline,
)


BASE_PIPELINE_MAP = {
    "StableDiffusionPipeline": StableDiffusionImg2ImgPipeline,
    "StableDiffusion3Pipeline": StableDiffusion3Img2ImgPipeline,
    "FluxPipeline": FluxImg2ImgPipeline,
    "StableDiffusionXLPipeline": StableDiffusionXLImg2ImgPipeline,
}


class ImageToImageNode(CustomNode):
    """Handles image-to-image generation using various models."""

    def __init__(self):
        super().__init__()
        self.config_manager = ModelConfigManager()
        self.model_memory_manager = get_model_memory_manager()

    async def _get_pipeline(self, model_id: str):
        pipeline = await self.model_memory_manager.load(model_id, None)
        if pipeline is None:
            raise ValueError(f"Model {model_id} not found in memory manager")
        return pipeline

    async def __call__(
        self,  # type: ignore
        source_image: Union[Image.Image, torch.Tensor, str],
        positive_prompt: str,
        model_id: str,
        negative_prompt: str = "",
        num_images: int = 1,
        strength: float = 0.8,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        try:
            pipeline = await self._get_pipeline(model_id)
            class_name = pipeline.__class__.__name__

            # Get model configuration
            model_config = self.config_manager.get_model_config(model_id, class_name)

            if not isinstance(source_image, list):
                source_image = [source_image]  # OmniGen expects a list of images/paths

            if class_name == "OmniGenPipeline":
                # Parameters specific to OmniGen pipeline
                gen_params = {
                    "prompt": positive_prompt,
                    "input_images": source_image,
                    # "guidance_scale": guidance_scale or model_config.get("guidance_scale", 3.0),
                    # "num_inference_steps": num_inference_steps or model_config.get("num_inference_steps", 50),
                    "use_img_guidance": True,
                    "img_guidance_scale": 1.6,
                    "output_type": "pt",
                    "seed": random_seed,
                    # "use_input_image_size_as_output": True
                    "return_dict": True,
                }
            else:
                # Parameters for standard diffusers pipelines
                # Set up generator for random seed
                generator = (
                    torch.Generator().manual_seed(random_seed)
                    if random_seed is not None
                    else None
                )
                img2img_pipeline = BASE_PIPELINE_MAP[class_name]
                pipeline = img2img_pipeline.from_pipe(pipeline)
                gen_params = {
                    "prompt": positive_prompt,
                    "negative_prompt": negative_prompt,
                    "image": source_image,
                    "strength": strength,
                    "num_inference_steps": num_inference_steps
                    or model_config.get("num_inference_steps", 50),
                    "guidance_scale": guidance_scale
                    or model_config.get("guidance_scale", 7.5),
                    "num_images_per_prompt": num_images,
                    "generator": generator,
                    "output_type": "pt",
                }

            # Generate images
            with torch.no_grad():
                output = pipeline(**gen_params)

            # Handle different output formats
            if isinstance(output, dict):
                # For OmniGen pipeline that returns dict
                images = output["images"]
            else:
                # For standard diffusers pipelines that return an object with .images
                images = output.images

            # Clean up memory
            self.model_memory_manager.flush_memory()

            # Ensure output is properly formatted
            if isinstance(images, list):
                images = torch.stack(images)
            elif images.dim() == 3:
                images = images.unsqueeze(0)

            return {"images": images}

        except Exception as e:
            raise RuntimeError(f"Error in image-to-image generation: {str(e)}")
