from typing import Dict, Optional, Union
import torch
from diffusers import (
    StableDiffusionPipeline, 
    DDIMScheduler,
    StableDiffusionXLPipeline as SDXLPipeline,
)
from core_library.node_interface import InputSpec, OutputSpec
from transformers import CLIPTokenizer

class CreatePipeline:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, InputSpec]:
        return {
            "vae": {
                "display_name": "VAE Model",
                "edge_type": "AutoencoderKL",
                "spec": {},
                "required": True,
            },
            "text_encoder": {
                "display_name": "Text Encoder",
                "edge_type": "CLIPTextModel",
                "spec": {},
                "required": True,
            },
            "text_encoder_2": {
                "display_name": "Text Encoder 2",
                "edge_type": "CLIPTextModel",
                "spec": {},
                "required": False,
            },
            "tokenizer": {
                "display_name": "Tokenizer",
                "edge_type": "CLIPTokenizer",
                "spec": {},
                "required": True,
            },
            "tokenizer_2": {
                "display_name": "Tokenizer 2",
                "edge_type": "CLIPTokenizer",
                "spec": {},
                "required": False,
            },
            "unet": {
                "display_name": "UNet Model",
                "edge_type": "UNet2DConditionModel",
                "spec": {},
                "required": True,
            },
            "scheduler": {
                "display_name": "Scheduler",
                "edge_type": "DDIMScheduler",
                "spec": {},
                "required": True,
            },
            "model_type": {
                "display_name": "Model Type",
                "edge_type": "str",
                "spec": {},
                "required": True,
            },
        }

    @property
    def RETURN_TYPES(self) -> OutputSpec:
        return {
            "display_name": "StableDiffusion Pipeline",
            "edge_type": "Union[StableDiffusionPipeline, SDXLPipeline]",
        }

    def __call__(
        self,
        vae,
        unet,
        text_encoder,
        text_encoder_2: Optional[torch.nn.Module] = None,
        tokenizer: Optional[CLIPTokenizer] = None,
        tokenizer_2: Optional[CLIPTokenizer] = None,    
        model_type: str = "sd1.5",
    ) -> Union[StableDiffusionPipeline, SDXLPipeline]:
        
        # Instantiate scheduler
        scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

        if model_type == "sdxl":
            return SDXLPipeline(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                unet=unet,
                scheduler=scheduler,
            )
        else:
            return StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
            )

    @property
    def CATEGORY(self) -> str:
        return "pipeline_creation"

    @property
    def display_name(self) -> Dict[str, str]:
        return {
            "en": "Create StableDiffusion Pipeline",
            "es": "Crear Pipeline de StableDiffusion",
        }

    @property
    def description(self) -> Dict[str, str]:
        return {
            "en": "Creates a StableDiffusion or SDXL pipeline from the loaded components.",
            "es": "Crea un pipeline de StableDiffusion o SDXL a partir de los componentes cargados.",
        }