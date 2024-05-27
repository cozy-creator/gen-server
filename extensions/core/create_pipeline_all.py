from typing import Dict, Optional, Union
import torch
from diffusers import (
    StableDiffusionPipeline, 
    DDIMScheduler,
    StableDiffusionXLPipeline as SDXLPipeline,
)
from core_library.node_interface import InputSpec, OutputSpec
from transformers import CLIPTokenizer
from core_library.model_handler import load_safetensors, load_unet, load_vae, load_text_encoder, load_text_encoder_2

class CreateStableDiffusionPipeline:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, InputSpec]:
        return {
            "safetensors_file": {
                "display_name": "Safetensors File",
                "edge_type": "str",
                "spec": {},
                "required": True,
            },
            "unet_config_file": {
                "display_name": "UNet Config File",
                "edge_type": "str",
                "spec": {},
                "required": True,
            },
            "vae_config_file": {
                "display_name": "VAE Config File",
                "edge_type": "str",
                "spec": {},
                "required": True,
            },
            "text_config_file": {
                "display_name": "Text Encoder Config File",
                "edge_type": "str",
                "spec": {},
                "required": True,
            },
            "text_2_config_file": {
                "display_name": "Text Encoder 2 Config File",
                "edge_type": "str",
                "spec": {},
                "required": False,
            },
            "model_type": {
                "display_name": "Model Type",
                "edge_type": "str",
                "spec": {},
                "required": True,
            },
            "device": {
                "display_name": "Device",
                "edge_type": "str",
                "spec": {},
                "required": False,
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
        safetensors_file: str,
        unet_config_file: str,
        vae_config_file: str,
        text_config_file: str,
        text_2_config_file: Optional[str] = None,
        model_type: str = "sd1.5",
        device: str = "cpu",
    ) -> Union[StableDiffusionPipeline, SDXLPipeline]:
        unet_state_dict = load_safetensors(safetensors_file, model_type, "unet")
        vae_state_dict = load_safetensors(safetensors_file, model_type, "vae")
        text_encoder_state_dict = load_safetensors(safetensors_file, model_type, "text_encoder")

        unet = load_unet(unet_state_dict, unet_config_file, device=device, model_type=model_type)
        vae = load_vae(vae_state_dict, vae_config_file, device=device, model_type=model_type)
        text_encoder = load_text_encoder(text_encoder_state_dict, text_config_file, device=device, model_type=model_type)

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

        if model_type == "sdxl":
            text_encoder_2_state_dict = load_safetensors(safetensors_file, model_type, "text_encoder_2")
            text_encoder_2 = load_text_encoder_2(
                text_encoder_2_state_dict, text_2_config_file, device=device, model_type=model_type, has_projection=True
            )
            tokenizer_2 = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

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
            "en": "Creates a StableDiffusion or SDXL pipeline by loading components from a Safetensors file and configuration files.",
            "es": "Crea un pipeline de StableDiffusion o SDXL cargando componentes desde un archivo de Safetensors y archivos de configuración.",
        }