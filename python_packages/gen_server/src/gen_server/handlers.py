from typing import List, Optional
from gen_server.globals import get_hf_model_manager, get_model_memory_manager
from gen_server.utils.model_config_manager import ModelConfigManager
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    FluxPipeline,
    # FluxControlNetModel,
    FluxControlNetPipeline,
    # FluxTransformer2DModel,
    # StableDiffusion3Pipeline,
)
import torch

models = {}

config_manager = ModelConfigManager()
hf_model_manager = get_hf_model_manager()
model_memory_manager = get_model_memory_manager()


async def load_model(model_id: str):
    if model_id not in models:
        pipeline = await model_memory_manager.load(model_id, None)
        if pipeline is None:
            raise ValueError(f"Model {model_id} not found in memory manager")
        models[model_id] = pipeline

    return model_id


def load_text_encoder(self, model_id: str):
    pass


async def generate_images(model_id: str, data: dict[str, any]):
    pipeline = models[model_id]
    if pipeline is None:
        raise ValueError(f"Model {model_id} not loaded")

    model_config = config_manager.get_model_config(
        model_id, pipeline.__class__.__name__
    )

    generator = None
    random_seed = data.get("random_seed")
    if random_seed is not None:
        generator = torch.Generator().manual_seed(random_seed)

    gen_params = {
        "num_inference_steps": model_config["num_inference_steps"],
        "num_images_per_prompt": data["num_images"],
        "prompt": data["positive_prompt"],
        "height": data["height"],
        "width": data["width"],
        "generator": generator,
        "output_type": "pt",
    }

    if isinstance(pipeline, FluxPipeline):
        gen_params["guidance_scale"] = 0.0
        gen_params["max_sequence_length"] = 256
    else:
        gen_params["negative_prompt"] = data["negative_prompt"]
        gen_params["guidance_scale"] = model_config["guidance_scale"]

    output = pipeline(**gen_params).images

    return output


def apply_controlnet(
    self,
    pipeline: DiffusionPipeline,
    controlnet_model_ids: List[str],
):
    class_name = pipeline.__class__.__name__

    controlnets = []
    _controlnet_inputs = []

    # for model_id in controlnet_model_ids:
    # if "openpose" in model_id.lower():
    #     controlnets.append(self._get_controlnet(model_id, "openpose", class_name))
    #     controlnet_inputs.append(openpose_image)
    # elif "depth" in model_id.lower():
    #     controlnets.append(self._get_controlnet(model_id, "depth", class_name))
    #     controlnet_inputs.append(depth_map)

    if isinstance(pipeline, (StableDiffusionPipeline, StableDiffusionXLPipeline)):
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
            pipeline, controlnet=controlnets, torch_dtype=torch.float16
        )
    # else:
    #     pipeline.controlnet = controlnets


def _handle_lora(self, pipeline: DiffusionPipeline, lora_info: Optional[dict]):
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
        pipeline.set_adapters(adapter_name, adapter_weights=[lora_info["model_scale"]])
        # pipeline.fuse_lora(lora_scale=lora_info["model_scale"], adapter_name=adapter_name)


def _get_controlnet(self, model_id: str, controlnet_type: str, class_name: str):
    key = f"{model_id}_{controlnet_type}"
    if key not in self.controlnets:
        if class_name == "StableDiffusionXLPipeline":
            if controlnet_type == "openpose":
                controlnet_id = "xinsir/controlnet-openpose-sdxl-1.0"
            elif controlnet_type == "depth":
                controlnet_id = "diffusers/controlnet-depth-sdxl-1.0"
        elif controlnet_type == "openpose":
            controlnet_id = "lllyasviel/control_v11p_sd15_openpose"
        elif controlnet_type == "depth":
            controlnet_id = "lllyasviel/sd-controlnet-depth"

        variants = ["bf16", "fp8", "fp16", None]  # None represents no variant

        for variant in variants:
            try:
                if variant is None:
                    self.controlnets[key] = ControlNetModel.from_pretrained(
                        controlnet_id, torch_dtype=torch.float16
                    )
                else:
                    self.controlnets[key] = ControlNetModel.from_pretrained(
                        controlnet_id, torch_dtype=torch.float16, variant=variant
                    )

                print(
                    f"\n\nControlNet {controlnet_id} loaded successfully with variant {variant}\n\n"
                )
                break
            except Exception:
                continue

        # self.controlnets[key] = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
    return self.controlnets[key]
