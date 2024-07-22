import torch


from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_edm_dpmsolver_multistep import EDMDPMSolverMultistepScheduler

from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
)
from typing import Optional
from PIL import Image
import os
import json

from gen_server.utils import load_models
from gen_server.utils.image import aspect_ratio_to_dimensions
from gen_server.base_types import CustomNode


class ImageGenNode(CustomNode):
    """Generates images using Stable Diffusion pipelines."""

    def __call__(
        self,
        checkpoint_id: str,
        positive_prompt: str,
        negative_prompt: str = "",
        aspect_ratio: str = "1/1",
        num_images: int = 1,
        random_seed: Optional[int] = None,
        checkpoint_files: dict = {},
        architectures: dict = {},
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            checkpoint_id: ID of the pre-trained model checkpoint to use.
            positive_prompt: Text prompt describing the desired image.
            negative_prompt: Text prompt describing what to avoid in the image.
            width: Output image width.
            height: Output image height.
            num_images: Number of images to generate.
            guidance_scale: CFG scale value.
            num_inference_steps: Number of inference steps.
            random_seed: Random seed for reproducibility (optional).
        Returns:
            A dictionary containing the list of generated PIL Images.
        """

        try:
            checkpoint_metadata = checkpoint_files.get(checkpoint_id, None)
            if checkpoint_metadata is None:
                raise ValueError(f"No checkpoint file found for ID: {checkpoint_id}")

            components = load_models.from_file(
                checkpoint_metadata.file_path, registry=architectures
            )

            match checkpoint_metadata.category:
                case "SD1":
                    pipe = self.create_sd1_pipe(components)
                    cfg = 7.0
                    num_inference_steps = 25
                    
                case "SDXL":
                    sdxl_type = checkpoint_metadata.components["core_extension_1.vae"][
                        "input_space"
                    ].lower()
                    pipe = self.create_sdxl_pipe(components, model_type=sdxl_type)
                    cfg = 7.0
                    num_inference_steps = 20
                    
                case "SD3":
                    pipe = self.create_sd3_pipe(components)
                    cfg = 7.0
                    num_inference_steps = 28
                    
                case _:
                    raise ValueError(
                        f"Unknown category: {checkpoint_metadata.category}"
                    )
                    
            # Determine the width and height based on the aspect ratio and base model
            width, height = aspect_ratio_to_dimensions(
                aspect_ratio, checkpoint_metadata.category
            )

            # More efficient dtype
            try:
                pipe.to(torch.bfloat16)
            except:
                pass

            # Optional Efficiency gains.
            # Enable_xformers_memory_efficient_attention can save memory usage and increase inference speed.
            # enable_model_cpu_offload and enable_vae_tiling can save memory usage.
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()

            # Run the pipeline
            generator = (
                torch.Generator(device="cpu").manual_seed(random_seed)
                if random_seed
                else None
            )
            
            tensor_images = pipe(
                prompt=positive_prompt,
                # negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_images_per_prompt=num_images,
                guidance_scale=cfg,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pt",
            ).images

            del pipe
            
            return { "images": tensor_images }

        except Exception as e:
            raise ValueError(f"Error generating images: {e}")

    def create_sd1_pipe(self, components: dict) -> StableDiffusionPipeline:
        vae = components["core_extension_1.vae"].model
        unet = components["core_extension_1.sd1_unet"].model
        text_encoder = components["core_extension_1.sd1_text_encoder"].model

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )

        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

        return pipe

    def create_sdxl_pipe(
        self, components: dict, model_type: Optional[str] = None
    ) -> StableDiffusionXLPipeline:
        vae = components["core_extension_1.vae"].model
        unet = components["core_extension_1.sdxl_unet"].model
        text_encoder_1 = components["core_extension_1.sdxl_text_encoder_1"].model
        text_encoder_2 = components["core_extension_1.text_encoder_2"].model

        if model_type == "playground":
            tokenizer = CLIPTokenizer.from_pretrained(
                "playgroundai/playground-v2.5-1024px-aesthetic", subfolder="tokenizer"
            )
            scheduler = EDMDPMSolverMultistepScheduler.from_pretrained(
                "playgroundai/playground-v2.5-1024px-aesthetic", subfolder="scheduler"
            )
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                "playgroundai/playground-v2.5-1024px-aesthetic", subfolder="tokenizer_2"
            )
        else:
            tokenizer = CLIPTokenizer.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer"
            )
            scheduler = EulerDiscreteScheduler.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
            )
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2"
            )

        pipe = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
        )

        return pipe

    def create_sd3_pipe(self, components: dict) -> StableDiffusion3Pipeline:
        vae = components["core_extension_1.vae"].model
        unet = components["core_extension_1.sd3_unet"].model
        text_encoder_1 = components["core_extension_1.sd3_text_encoder_1"].model
        text_encoder_2 = components["core_extension_1.text_encoder_2"].model
        text_encoder_3 = components["core_extension_1.sd3_text_encoder_3"].model

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        )
        tokenizer_3 = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="scheduler"
        )
        # scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        print("Started SD3")
        pipe = StableDiffusion3Pipeline(
            vae=vae,
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=unet,
            scheduler=scheduler,
        )

        print("Done!")

        return pipe

    @staticmethod
    def get_spec():
        """Returns the node specification."""
        spec_file = os.path.join(os.path.dirname(__file__), "image_gen_node.json")
        with open(spec_file, "r", encoding="utf-8") as f:
            spec = json.load(f)
        return spec
