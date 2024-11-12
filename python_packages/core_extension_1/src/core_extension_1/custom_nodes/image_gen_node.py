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
from typing import Callable, Optional, List

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
    StableDiffusion3Pipeline,
)

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
from gen_server.utils.image import tensor_to_pil
from tqdm import tqdm


# from gen_server.utils.model_memory_manager import ModelMemoryManager


poseable_character_task_data = {
    # Models information
    "models": {
        "stabilityai/stable-diffusion-xl-base-1.0": 1,
        "runwayml/stable-diffusion-v1-5": 2,
    },
    # Image generation parameters
    "positive_prompt": "A beautiful anime girl with long blue hair, wearing a white dress, standing in a field of flowers",
    "negative_prompt": "low quality, bad anatomy, distorted face, blurry",
    "aspect_ratio": "3/4",
    "random_seed": 42,
    # ControlNet inputs
    "pose_image": "path/to/pose_reference.png",
    "depth_image": "path/to/depth_reference.png",
    "controlnet_model_ids": [
        "xinsir/controlnet-openpose-sdxl-1.0",
        "diffusers/controlnet-depth-sdxl-1.0",
    ],
    # IP-Adapter input
    "style_image": "path/to/style_reference.jpg",
    # LoRA information (optional)
    "lora_info": {
        "repo_id": "path/to/lora/weights",
        "weight_name": "anime_style_lora.safetensors",
        "adapter_name": "anime_style",
        "model_scale": 0.7,
        "text_encoder_scale": 0.5,
        "text_encoder_2_scale": 0.5,
    },
    # Face regeneration parameters
    "face_prompt": "beautiful anime face, detailed eyes, soft smile",
    "face_mask_feather_iterations": 5,
    # Background generation parameters
    "background_prompt": "Beautiful flower field, sunny day, anime style background",
    # Additional parameters
    "guidance_scale": 7.5,
    "num_inference_steps": 30,
    "strength": 0.7,  # For face regeneration
}


# images_completed = overall_step // self.num_steps
        
# if images_completed < self.num_images:
#     self.pbar.set_postfix({
#         "Image": f"{images_completed}/{self.num_images}"
#     })

class ProgressCallback:
    def __init__(self, num_steps, num_images):
        self.num_steps = num_steps
        self.num_images = num_images
        self.total_steps = num_steps * num_images
        self.pbar = tqdm(total=num_steps, desc="Generating images")
        self.last_update = 0

    def on_step_end(self, pipeline, step_number, timestep, callback_kwargs):
        overall_step = self.last_update + step_number
        scaled_step = int((overall_step / self.total_steps) * self.num_steps)
        
        if scaled_step > self.pbar.n:
            self.pbar.update(scaled_step - self.pbar.n)

        # Return the latents unchanged
        return {"latents": callback_kwargs["latents"]}

    def on_image_complete(self):
        self.last_update += self.num_steps

    def close(self):
        self.pbar.n = self.num_steps
        self.pbar.refresh()
        self.pbar.close()


class ImageGenNode(CustomNode):
    """Generates images using Stable Diffusion pipelines."""

    def __init__(self):
        super().__init__()
        self.config_manager = ModelConfigManager()
        self.model_memory_manager = get_model_memory_manager()
        self.hf_model_manager = get_hf_model_manager()
        self.controlnets = {}

    async def _get_pipeline(self, model_id: str):
        pipeline = await self.model_memory_manager.load(model_id, None)
        if pipeline is None:
            raise ValueError(f"Model {model_id} not found in memory manager")
        return pipeline

    def _get_controlnet(self, model_id: str, controlnet_type: str, class_name: str):
        key = f"{model_id}_{controlnet_type}"
        if key not in self.controlnets:
            if class_name == "StableDiffusionXLPipeline":
                if controlnet_type == "openpose":
                    controlnet_id = "xinsir/controlnet-openpose-sdxl-1.0"
                elif controlnet_type == "depth":
                    controlnet_id = "diffusers/controlnet-depth-sdxl-1.0"
            elif class_name == "FluxPipeline":
                controlnet_id = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"
            else:
                if controlnet_type == "openpose":
                    controlnet_id = "lllyasviel/control_v11p_sd15_openpose"
                elif controlnet_type == "depth":
                    controlnet_id = "lllyasviel/sd-controlnet-depth"

            variants = ["bf16", "fp8", "fp16", None]  # None represents no variant

            if class_name == "FluxPipeline":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16

            for variant in variants:
                try:
                    if variant is None:
                        self.controlnets[key] = ControlNetModel.from_pretrained(
                            controlnet_id, torch_dtype=torch_dtype
                        )
                    else:
                        self.controlnets[key] = ControlNetModel.from_pretrained(
                            controlnet_id, torch_dtype=torch_dtype, variant=variant
                        )

                    print(
                        f"\n\nControlNet {controlnet_id} loaded successfully with variant {variant}\n\n"
                    )
                    break
                except Exception as e:
                    continue

            # self.controlnets[key] = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
        return self.controlnets[key]

    async def __call__(  # type: ignore
        self,
        model_id: str,
        positive_prompt: str,
        negative_prompt: str = "",
        aspect_ratio: str = "1/1",
        num_images: int = 1,
        random_seed: Optional[int] = None,
        openpose_image: Optional[torch.Tensor] = None,
        depth_map: Optional[torch.Tensor] = None,
        ip_adapter_embeds: Optional[torch.Tensor] = None,
        lora_info: Optional[dict[str, any]] = None,
        controlnet_model_ids: Optional[List[str]] = None,
    ):
        
        print(random_seed)
        try:
            pipeline = await self._get_pipeline(model_id)

            if pipeline is None:
                return None

            # repo_id = pipeline._name_or_path

            class_name = pipeline.__class__.__name__

            model_config = self.config_manager.get_model_config(model_id, class_name)

            # Already handled by the method
            if lora_info:
                self.handle_lora(pipeline, lora_info)
                print("DONE HANDLING LORA")

            controlnet_inputs = []
            if controlnet_model_ids:
                print(f"Controlnet model ids: {controlnet_model_ids}")
                controlnets = []
                for model_id in controlnet_model_ids:
                    if "openpose" in model_id.lower():
                        controlnets.append(
                            self._get_controlnet(model_id, "openpose", class_name)
                        )
                        controlnet_inputs.append(openpose_image)
                    elif "depth" in model_id.lower():
                        controlnets.append(
                            self._get_controlnet(model_id, "depth", class_name)
                        )
                        controlnet_inputs.append(depth_map)

                if isinstance(
                    pipeline, (StableDiffusionPipeline, StableDiffusionXLPipeline)
                ):
                    if class_name == "StableDiffusionXLPipeline":
                        pipeline = StableDiffusionXLControlNetPipeline.from_pipe(
                            pipeline, controlnet=controlnets, torch_dtype=torch.float16
                        )
                    else:
                        pipeline = StableDiffusionControlNetPipeline.from_pipe(
                            pipeline, controlnet=controlnets, torch_dtype=torch.float16
                        )
                elif isinstance(pipeline, FluxPipeline):
                    pipeline = FluxControlNetPipeline.from_pipe(
                        pipeline, controlnet=controlnets, torch_dtype=torch.bfloat16
                    )
                # else:
                #     pipeline.controlnet = controlnets

            if ip_adapter_embeds is not None:
                self.setup_ip_adapter(pipeline, model_config["category"])

                print("DONE SETTING UP IP ADAPTER")


            width, height = aspect_ratio_to_dimensions(aspect_ratio, class_name)

            gen_params = {
                "prompt": positive_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": model_config["num_inference_steps"],
                "generator": torch.Generator().manual_seed(random_seed)
                if random_seed is not None
                else None,
                "output_type": "pt",
            }

            if isinstance(pipeline, FluxPipeline):
                gen_params["max_sequence_length"] = model_config["max_sequence_length"]
            else:
                if model_id != "playground2.5":
                    # TODO: this is a temporary fix to remove the negative prompt. Ensure to add it back in when the frontend is working.
                    gen_params["negative_prompt"] = negative_prompt

            gen_params["guidance_scale"] = model_config["guidance_scale"]

            if controlnet_inputs:
                gen_params["image"] = controlnet_inputs

            if ip_adapter_embeds is not None:
                gen_params["ip_adapter_image_embeds"] = ip_adapter_embeds

            # Initialize an empty list to store the image tensors
            image_tensors = []

            pipeline.set_progress_bar_config(disable=True)
            callback = ProgressCallback(model_config["num_inference_steps"], num_images)

            # Run inference
            for i in range(num_images):
                with torch.no_grad():
                    output = pipeline(
                        **gen_params, 
                        callback_on_step_end=callback.on_step_end, 
                        callback_on_step_end_tensor_inputs=['latents']
                    ).images

                image_tensors.append(output[0]) 
                # Clear CUDA cache after each iteration
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                callback.on_image_complete() # Signal completion of an image


            callback.close()  # Close the progress bar

            all_images = torch.stack(image_tensors)

            return {"images": all_images}
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Error generating images: {e}")

    def setup_controlnet(self, pipeline: any, controlnet_info: dict):
        if isinstance(pipeline, FluxPipeline):
            controlnet = FluxControlNetModel.from_pretrained(
                controlnet_info["model_id"]
            )

            new_pipeline = FluxControlNetPipeline.from_pipe(
                pipeline, controlnet=controlnet
            )

            return new_pipeline

        controlnet = ControlNetModel.from_pretrained(
            controlnet_info["model_id"], torch_dtype=torch.float16
        )

        if isinstance(pipeline, StableDiffusionXLPipeline):
            new_pipeline = StableDiffusionXLControlNetPipeline.from_pipe(
                pipeline, controlnet=controlnet
            )

        elif isinstance(pipeline, StableDiffusionPipeline):
            new_pipeline = StableDiffusionControlNetPipeline.from_pipe(
                pipeline, controlnet=controlnet
            )

        else:
            raise ValueError(
                f"Unsupported pipeline type for ControlNet: {type(pipeline)}"
            )

        return new_pipeline

    def handle_lora(self, pipeline: DiffusionPipeline, lora_info: dict = None):
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
                    adapter_name=adapter_name,
                )
                print(f"LoRA adapter '{adapter_name}' loaded successfully.")
            except ValueError as e:
                if "already in use" in str(e):
                    print(
                        f"LoRA adapter '{adapter_name}' is already loaded. Using existing adapter."
                    )

                else:
                    raise e

            # Set LoRA scales
            lora_scale_dict = {}

            if hasattr(pipeline, "text_encoder"):
                lora_scale_dict["text_encoder"] = lora_info["text_encoder_scale"]
            if hasattr(pipeline, "text_encoder_2"):
                lora_scale_dict["text_encoder_2"] = lora_info["text_encoder_2_scale"]

            # Determine if the model uses UNet or Transformer
            if hasattr(pipeline, "unet"):
                lora_scale_dict["unet"] = lora_info["model_scale"]
            elif hasattr(pipeline, "transformer"):
                lora_scale_dict["transformer"] = lora_info["model_scale"]

            # Set the scales
            pipeline.set_adapters(
                adapter_name, adapter_weights=[lora_info["model_scale"]]
            )
            # pipeline.fuse_lora(lora_scale=lora_info["model_scale"], adapter_name=adapter_name)

    def setup_ip_adapter(self, pipeline: DiffusionPipeline, model_category: str):
        if model_category == "sdxl":
            pipeline.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.bin",
            )
        else:
            pipeline.load_ip_adapter(
                "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin"
            )
        pipeline.set_ip_adapter_scale(0.7)
