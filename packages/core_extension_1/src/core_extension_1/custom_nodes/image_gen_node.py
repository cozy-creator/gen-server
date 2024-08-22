import traceback
import torch

from diffusers.pipelines.pipeline_utils import DiffusionPipeline

# from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
# from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
# from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
# from diffusers.schedulers.scheduling_ddim import DDIMScheduler
# from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
# from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
# from diffusers.schedulers.scheduling_edm_dpmsolver_multistep import EDMDPMSolverMultistepScheduler
from typing import Callable, Optional

from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    FluxPipeline,
    FluxControlNetModel,
    FluxControlNetPipeline,
    FluxTransformer2DModel,
    StableDiffusion3Pipeline
)

StableDiffusion3Pipeline.load_lora_weights

from gen_server.utils.image import aspect_ratio_to_dimensions
from gen_server.base_types import CustomNode
from importlib import import_module
from gen_server.utils.model_config_manager import ModelConfigManager
from gen_server.globals import (
    get_hf_model_manager,
    get_available_torch_device,
    get_model_memory_manager,
)
from diffusers import FluxPipeline
import os
import json


# from gen_server.utils.model_memory_manager import ModelMemoryManager


class ImageGenNode(CustomNode):
    """Generates images using Stable Diffusion pipelines."""

    def __init__(self):
        super().__init__()
        self.config_manager = ModelConfigManager()
        self.model_memory_manager = get_model_memory_manager()
        self.hf_model_manager = get_hf_model_manager()

    async def __call__( # type: ignore
        self,
        repo_id: str,
        positive_prompt: str,
        negative_prompt: str = "",
        aspect_ratio: str = "1/1",
        num_images: int = 1,
        random_seed: Optional[int] = None,
        callback: Optional[Callable] = None,
        callback_steps: Optional[int] = 1,
        lora_info: Optional[dict[str, any]] = None,
        controlnet_info: Optional[dict[str, any]] = None,
    ):
        try:
            pipeline = await self.model_memory_manager.load(repo_id, None)
            if pipeline is None:
                return None

            class_name = pipeline.__class__.__name__
            module = import_module("diffusers")

            model_config = self.config_manager.get_model_config(repo_id, class_name)

            scheduler_name = model_config.get("scheduler")
            if scheduler_name:
                SchedulerClass = getattr(module, scheduler_name)
                pipeline.scheduler = SchedulerClass.from_config(pipeline.scheduler.config)

            width, height = aspect_ratio_to_dimensions(aspect_ratio, class_name)


            if isinstance(pipeline, (DiffusionPipeline)):
                self.handle_lora(pipeline, lora_info)

            if controlnet_info:
                pipeline = self.setup_controlnet(pipeline, controlnet_info)

            self.model_memory_manager.apply_optimizations(pipeline)

            full_positive_prompt = f"{model_config['default_positive_prompt']} {positive_prompt}".strip()
            full_negative_prompt = f"{model_config['default_negative_prompt']} {negative_prompt}".strip()

            gen_params = {
                "prompt": full_positive_prompt,
                "negative_prompt": full_negative_prompt,
                "width": width,
                "height": height,
                "num_images_per_prompt": num_images,
                "num_inference_steps": model_config["num_inference_steps"],
                "generator": torch.Generator().manual_seed(random_seed) if random_seed is not None else None,
                "output_type": "pt",
                "callback_on_step_end": callback,
                # "callback_steps": callback_steps,
            }

            if isinstance(pipeline, FluxPipeline):
                gen_params["guidance_scale"] = 0.0
                gen_params["max_sequence_length"] = 256
                # Remove negative_prompt for Flux
                gen_params.pop("negative_prompt", None)
            else:
                gen_params["guidance_scale"] = model_config["guidance_scale"]

            if controlnet_info:
                gen_params["image"] = controlnet_info.get("control_image")
                gen_params["controlnet_conditioning_scale"] = controlnet_info.get("conditioning_scale", 1.0)
                if "guess_mode" in controlnet_info:
                    gen_params["guess_mode"] = controlnet_info["guess_mode"]

            tensor_images = pipeline(**gen_params).images

            return {"images": tensor_images}
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Error generating images: {e}")
        
    def setup_controlnet(self, pipeline: any, controlnet_info: dict):

        if isinstance(pipeline, FluxPipeline):
            controlnet = FluxControlNetModel.from_pretrained(
                controlnet_info["model_id"]
            )

            new_pipeline = FluxControlNetPipeline.from_pipe(
                pipeline,
                controlnet=controlnet
            )

            return new_pipeline


        controlnet = ControlNetModel.from_pretrained(
            controlnet_info["model_id"],
            torch_dtype=torch.float16
        )

        if isinstance(pipeline, StableDiffusionXLPipeline):
            new_pipeline = StableDiffusionXLControlNetPipeline.from_pipe(
                pipeline,
                controlnet=controlnet
            )
        elif isinstance(pipeline, StableDiffusionPipeline):
            new_pipeline = StableDiffusionControlNetPipeline.from_pipe(
                pipeline,
                controlnet=controlnet
            )

        else:
            raise ValueError(f"Unsupported pipeline type for ControlNet: {type(pipeline)}")


        return new_pipeline
    
    def handle_lora(self, pipeline: DiffusionPipeline, lora_info: dict):
        if lora_info is None:
            # If no LoRA info is provided, disable all LoRAs
            pipeline.unload_lora_weights()
            
        else:
            print("Loading LoRA weights...")
            adapter_name = lora_info["adapter_name"]
            print(f"Adapter Name: {adapter_name}")
            try:
                pipeline.load_lora_weights(
                    lora_info["repo_id"],
                    weight_name=lora_info["weight_name"],
                    adapter_name=adapter_name
                )
                print(f"LoRA adapter '{adapter_name}' loaded successfully.")
            except ValueError as e:
                if "already in use" in str(e):
                    print(f"LoRA adapter '{adapter_name}' is already loaded. Using existing adapter.")
                    
                else:
                    raise e

            # Set LoRA scales
            lora_scale_dict = {}

            if hasattr(pipeline, 'text_encoder'):
                lora_scale_dict["text_encoder"] = lora_info["text_encoder_scale"]
            if hasattr(pipeline, 'text_encoder_2'):
                lora_scale_dict["text_encoder_2"] = lora_info["text_encoder_2_scale"]

            # Determine if the model uses UNet or Transformer
            if hasattr(pipeline, 'unet'):
                lora_scale_dict["unet"] = lora_info["model_scale"]
            elif hasattr(pipeline, 'transformer'):
                lora_scale_dict["transformer"] = lora_info["model_scale"]

            # Set the scales
            pipeline.set_adapters(adapter_name, adapter_weights=[lora_info["model_scale"]])
            # pipeline.fuse_lora(lora_scale=lora_info["model_scale"], adapter_name=adapter_name)

    # @staticmethod
    # def get_spec():
    #     """Returns the node specification."""
    #     spec_file = os.path.join(os.path.dirname(__file__), 'audio_node.json')
    #     with open(spec_file, 'r', encoding='utf-8') as f:
    #         spec = json.load(f)
    #     return spec


#     def apply_optimizations(self, pipeline: DiffusionPipeline):
#         device = get_available_torch_device()
#         optimizations = [
#             ("enable_vae_tiling", "VAE Tiled", {}),
#             (
#                 "enable_xformers_memory_efficient_attention",
#                 "Memory Efficient Attention",
#                 {},
#             ),
#             (
#                 "enable_model_cpu_offload",
#                 "CPU Offloading",
#                 {"device": device},
#             ),
#         ]

#         # Patch torch.mps to torch.backends.mps
#         device_type = device if isinstance(device, str) else device.type
#         if device_type == "mps":
#             setattr(torch, "mps", torch.backends.mps)
#         for opt_func, opt_name, kwargs in optimizations:
#             try:
#                 getattr(pipeline, opt_func)(**kwargs)
#                 print(f"{opt_name} enabled")
#             except Exception as e:
#                 print(f"Error enabling {opt_name}: {e}")

#         delattr(torch, "mps")


# class ImageGenNode(CustomNode):
#     """Generates images using Stable Diffusion pipelines."""

#     def __call__(
#         self,
#         checkpoint_id: str,
#         positive_prompt: str,
#         negative_prompt: str = "",
#         aspect_ratio: str = "1/1",
#         num_images: int = 1,
#         random_seed: Optional[int] = None,
#         checkpoint_files: dict = {},
#         architectures: dict = {},
#         device: torch.device ="cuda",
#     ) -> dict[str, torch.Tensor]:
#         """
#         Args:
#             checkpoint_id: ID of the pre-trained model checkpoint to use.
#             positive_prompt: Text prompt describing the desired image.
#             negative_prompt: Text prompt describing what to avoid in the image.
#             width: Output image width.
#             height: Output image height.
#             num_images: Number of images to generate.
#             guidance_scale: CFG scale value.
#             num_inference_steps: Number of inference steps.
#             random_seed: Random seed for reproducibility (optional).
#         Returns:
#             A dictionary containing the list of generated PIL Images.
#         """

#         try:
#             checkpoint_metadata = checkpoint_files.get(checkpoint_id, None)
#             if checkpoint_metadata is None:
#                 raise ValueError(f"No checkpoint file found for ID: {checkpoint_id}")

#             components = load_models.from_file(
#                 checkpoint_metadata.file_path,
#                 registry=architectures,
#                 # device=device,
#             )

#             print("Checkpoint Metadata", checkpoint_metadata)

#             match checkpoint_metadata.category:
#                 case "SD1":
#                     pipe = self.create_sd1_pipe(components)
#                     cfg = 7.0
#                     num_inference_steps = 25

#                 case "SDXL":
#                     sdxl_type = checkpoint_metadata.components["core_extension_1.vae"][
#                         "input_space"
#                     ].lower()
#                     pipe = self.create_sdxl_pipe(components, model_type=sdxl_type)
#                     cfg = 7.0
#                     num_inference_steps = 20

#                 case "SD3":
#                     pipe = self.create_sd3_pipe(components)
#                     cfg = 7.0
#                     num_inference_steps = 28

#                 case "AuraFlow":
#                     pipe = self.create_auraflow_pipe(components)
#                     cfg = 3.5
#                     num_inference_steps = 20

#                 case _:
#                     raise ValueError(
#                         f"Unknown category: {checkpoint_metadata.category}"
#                     )

#             # Determine the width and height based on the aspect ratio and base model
#             width, height = aspect_ratio_to_dimensions(
#                 aspect_ratio, checkpoint_metadata.category
#             )

#             # More efficient dtype
#             # try:
#             #     print(f"Device: {device}")
#             #     pipe.to(device=device, dtype=torch.bfloat16)
#             #     print(f"Pipe: {pipe.device}, DType: {pipe.dtype}")
#             # except:
#             #     pass

#             # Optional Efficiency gains.
#             # Enable_xformers_memory_efficient_attention can save memory usage and increase inference speed.
#             # enable_model_cpu_offload and enable_vae_tiling can save memory usage.

#             if not checkpoint_metadata.category == "AuraFlow":
#                 try:
#                     pipe.enable_vae_tiling()
#                     print("VAE Tiled")
#                 except Exception as e:
#                     print("error here.... ")
#                     print(f"Error: {e}")
#                 try:
#                     pipe.enable_xformers_memory_efficient_attention()
#                     print("Memory Efficient Attention")
#                 except Exception as e:
#                     print(f"Error: {e}")

#             pipe.enable_model_cpu_offload()
#             print("Done offloading")

#             # Run the pipeline
#             generator = (
#                 torch.Generator().manual_seed(random_seed) if random_seed else None
#             )

#             tensor_images = pipe(
#                 prompt=positive_prompt,
#                 negative_prompt=negative_prompt,
#                 width=width,
#                 height=height,
#                 num_images_per_prompt=num_images,
#                 guidance_scale=cfg,
#                 num_inference_steps=num_inference_steps,
#                 generator=generator,
#                 output_type="pt",
#             ).images

#             print("I'm done")

#             #     image = pipe(
#             #     prompt="A person with green hair standing at the left side of the screen, a man with blue hair at the centre and a baby with red hair at the right playing with a toy. There is a yellow UFO looming over them.",
#             #     height=1024,
#             #     width=1024,
#             #     num_images_per_prompt=1,
#             #     num_inference_steps=20,
#             #     generator=torch.Generator().manual_seed(234),
#             #     guidance_scale=3.5,
#             #     output_type="pt"
#             # ).images

#             del pipe

#             return {"images": tensor_images}

#         except Exception as e:
#             traceback.print_exc()
#             raise ValueError(f"Error generating images: {e}")

#     def create_sd1_pipe(self, components: dict) -> StableDiffusionPipeline:
#         vae = components["core_extension_1.sd1_vae"].model
#         unet = components["core_extension_1.sd1_unet"].model
#         text_encoder = components["core_extension_1.sd1_text_encoder"].model

#         tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
#         scheduler = DDIMScheduler.from_pretrained(
#             "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
#         )

#         pipe = StableDiffusionPipeline(
#             vae=vae,
#             text_encoder=text_encoder,
#             tokenizer=tokenizer,
#             unet=unet,
#             scheduler=scheduler,
#             safety_checker=None,
#             feature_extractor=None,
#             requires_safety_checker=False,
#         )

#         return pipe

#     def create_sdxl_pipe(
#         self, components: dict, model_type: Optional[str] = None
#     ) -> StableDiffusionXLPipeline:
#         vae = components["core_extension_1.vae"].model
#         unet = components["core_extension_1.sdxl_unet"].model
#         text_encoder_1 = components["core_extension_1.sdxl_text_encoder_1"].model
#         text_encoder_2 = components["core_extension_1.text_encoder_2"].model

#         if model_type == "playground":
#             tokenizer = CLIPTokenizer.from_pretrained(
#                 "playgroundai/playground-v2.5-1024px-aesthetic", subfolder="tokenizer"
#             )
#             scheduler = EDMDPMSolverMultistepScheduler.from_pretrained(
#                 "playgroundai/playground-v2.5-1024px-aesthetic", subfolder="scheduler"
#             )
#             tokenizer_2 = CLIPTokenizer.from_pretrained(
#                 "playgroundai/playground-v2.5-1024px-aesthetic", subfolder="tokenizer_2"
#             )
#         elif model_type == "pony":
#             print("In Pony")
#             tokenizer = CLIPTokenizer.from_pretrained(
#                 "stablediffusionapi/pony-realism", subfolder="tokenizer"
#             )
#             pipe_scheduler = EulerDiscreteScheduler.from_pretrained(
#                 "stablediffusionapi/pony-realism", subfolder="scheduler"
#             )

#             scheduler = DPMSolverMultistepScheduler.from_config(
#                     pipe_scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True
#                 )
#             tokenizer_2 = CLIPTokenizer.from_pretrained(
#                 "stablediffusionapi/pony-realism", subfolder="tokenizer_2"
#             )
#         else:
#             tokenizer = CLIPTokenizer.from_pretrained(
#                 "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer"
#             )
#             scheduler = EulerDiscreteScheduler.from_pretrained(
#                 "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
#             )
#             tokenizer_2 = CLIPTokenizer.from_pretrained(
#                 "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2"
#             )

#         pipe = StableDiffusionXLPipeline(
#             vae=vae,
#             text_encoder=text_encoder_1,
#             text_encoder_2=text_encoder_2,
#             unet=unet,
#             scheduler=scheduler,
#             tokenizer=tokenizer,
#             tokenizer_2=tokenizer_2,
#         )

#         return pipe

#     def create_sd3_pipe(self, components: dict) -> StableDiffusion3Pipeline:
#         vae = components["core_extension_1.sd3_vae"].model
#         unet = components["core_extension_1.sd3_unet"].model
#         text_encoder_1 = components["core_extension_1.sd3_text_encoder_1"].model
#         text_encoder_2 = components["core_extension_1.text_encoder_2"].model
#         text_encoder_3 = components["core_extension_1.sd3_text_encoder_3"].model

#         tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
#         tokenizer_2 = CLIPTokenizer.from_pretrained(
#             "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
#         )
#         tokenizer_3 = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")

#         scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
#             "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="scheduler"
#         )
#         # scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

#         print("Started SD3")
#         pipe = StableDiffusion3Pipeline(
#             vae=vae,
#             text_encoder=text_encoder_1,
#             text_encoder_2=text_encoder_2,
#             text_encoder_3=text_encoder_3,
#             tokenizer=tokenizer,
#             tokenizer_2=tokenizer_2,
#             tokenizer_3=tokenizer_3,
#             transformer=unet,
#             scheduler=scheduler,
#         )

#         print("Done!")

#         return pipe

#     def create_auraflow_pipe(self, components: dict) -> AuraFlowPipeline:
#         vae = components["core_extension_1.auraflow_vae"].model
#         transformer = components["core_extension_1.auraflow_transformer"].model
#         text_encoder = components["core_extension_1.auraflow_text_encoder"].model

#         # Load scheduler and Tokenizer
#         scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
#             "fal/AuraFlow", subfolder="scheduler"
#         )
#         tokenizer = LlamaTokenizerFast.from_pretrained(
#             "fal/AuraFlow", subfolder="tokenizer"
#         )

#         pipe = AuraFlowPipeline(
#             vae=vae,
#             text_encoder=text_encoder,
#             tokenizer=tokenizer,
#             transformer=transformer,
#             scheduler=scheduler,
#         )

#         return pipe

#     @staticmethod
#     def get_spec():
#         """Returns the node specification."""
#         spec_file = os.path.join(os.path.dirname(__file__), "image_gen_node.json")
#         with open(spec_file, "r", encoding="utf-8") as f:
#             spec = json.load(f)
#         return spec
