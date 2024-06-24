import sys
from typing import Any, Union, List, Optional, Dict

# from RealESRGAN.rrdbnet_arch import RRDBNet

from gen_server.utils import load_models, components_from_state_dict
from gen_server.base_types import (
    Architecture,
    TorchDevice,
    NodeInterface,
    ModelConstraint,
    ImageOutputType,
    CustomNode,
    Language,
    Category,
)
from gen_server.globals import CHECKPOINT_FILES
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusion3Pipeline,
    DDIMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    SD3Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusionXLPipeline,
)
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    T5TokenizerFast,
    CLIPTextModelWithProjection,
    T5EncoderModel,
)

from core_extension_1.widgets import TextInput, StringInput, EnumInput
import json
import torch
import os
from PIL.Image import Image
from numpy import ndarray
from spandrel import ModelLoader

from image_utils.custom_nodes import pil_to_tensor, tensor_to_pil

# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp


config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scheduler_config.json"
)


# TO DO: 'device' should somehow be marked as an internal-only parameter
# reserved just for the executor to have tighter control. It should NOT
# be a regular input. We do not want the end-user to worry about the device
# they are running on.
#
# TO DO: DO NOT load all models! Only the ones needed!
class LoadCheckpoint(CustomNode):
    """
    Takes a file with a state dict, outputs a dictionary of model-classes from Diffusers
    """

    display_name = {
        Language.ENGLISH: "Load Checkpoint",
        Language.CHINESE: "加载检查点",
    }

    category = Category.LOADER

    description = {
        Language.ENGLISH: "Loads a checkpoint file and returns a dictionary of model-classes",
        Language.CHINESE: "加载检查点文件，并返回模型类的字典",
    }

    @staticmethod
    def update_interface(inputs: dict[str, Any]) -> NodeInterface:
        interface = {
            "inputs": {"file_path": EnumInput(options=list(CHECKPOINT_FILES.keys()))},
            "outputs": {},
        }
        if inputs:
            file_path = inputs.get("file_path", None)
            if file_path:
                interface.update({"outputs": components_from_state_dict(file_path)})

        return interface

    # TODO: do something without output-keys? maybe some more declarative
    def __call__(
        self, file_path: str, *, output_keys: dict = {}, device: TorchDevice = None
    ) -> dict[str, Architecture]:
        return load_models.from_file(file_path, device)


class CreatePipe(CustomNode):
    """
    Produces a diffusers pipeline, and loads it onto the device.
    """

    display_name = "Create Pipe"

    category = "pipe"

    description = "Creates a StableDiffusionPipeline"

    # Note the declarative nature of this; the custom-node declares what types
    # it should receive, and the executor handles the details of ensuring this.
    # custom nodes shouldn't have to spend much time validating / sanitizing their inputs
    # that should be the executor's job. The custom nodes should be declarative
    @staticmethod
    def update_interface(inputs: dict[str, Any] = None) -> NodeInterface:
        interface = {
            "inputs": {"unet": ModelConstraint(model_type=UNet2DConditionModel)},
            "outputs": {"pipe": StableDiffusionPipeline},
        }

        if (
            inputs is not None
            and isinstance(inputs.get("unet"), Architecture)
            and isinstance(inputs["unet"].model, UNet2DConditionModel)
        ):
            # Ensure that the vae and text_encoder are compatible with this unet
            arch: Architecture = inputs["unet"]
            interface["inputs"].update(
                {
                    "vae": ModelConstraint(
                        model_type=AutoencoderKL,
                        input_space=arch.input_space,
                        output_space=arch.output_space,
                    ),
                    "text_encoder": ModelConstraint(
                        model_type=CLIPTextModel, output_space=arch.input_space
                    ),
                }
            )

        return interface

    def __call__(
        self,
        unet: Union[UNet2DConditionModel, SD3Transformer2DModel],
        vae: AutoencoderKL,
        text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection],
        text_encoder_2: Optional[CLIPTextModelWithProjection] = None,
        text_encoder_3: Optional[T5EncoderModel] = None,
        device: Optional[TorchDevice] = None,
    ) -> Union[StableDiffusion3Pipeline, StableDiffusionXLPipeline]:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        if isinstance(unet, SD3Transformer2DModel) and text_encoder_3 is not None:
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
            )
            tokenizer_3 = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")
            # tokenizer_2 = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
            # scheduler = DDIMScheduler.from_pretrained(
            #     "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
            # )

            scheduler_config = json.load(open(config_path))
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

            pipe = StableDiffusion3Pipeline(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                text_encoder_3=text_encoder_3,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                tokenizer_3=tokenizer_3,
                transformer=unet,
                scheduler=scheduler,
            ).to(torch.bfloat16)
        elif text_encoder_2 is not None and text_encoder_3 is None:
            scheduler = DDIMScheduler.from_pretrained(
                "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
            )
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
            )
            pipe = StableDiffusionXLPipeline(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                unet=unet,
                scheduler=scheduler,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
            ).to(torch.bfloat16)
        else:
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

        # if "xformers" in sys.modules:
        #     pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        if "accelerate" in sys.modules:
            pipe.enable_model_cpu_offload()
        # pipe.enable_vae_tiling()
        pipe.to(device)

        return pipe


class RunPipe(CustomNode):
    """
    Takes a StableDiffusionPipeline and a prompt, outputs an image
    """

    display_name = "Run Pipe"

    category = "pipe"

    description = "Runs a StableDiffusionPipeline with a prompt"

    # TODO: this needs to be updated to support SDXL and other pipelines
    @staticmethod
    def update_interface(inputs: dict[str, Any] = None) -> NodeInterface:
        interface = {
            "inputs": {
                "pipe": StableDiffusionPipeline,
                "negative_prompt": TextInput(),
                "prompt": TextInput(),
            },
            "outputs": {"image_output": ImageOutputType},
        }

        return interface

    def __call__(
        self,
        pipe: StableDiffusionPipeline,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Union[List[Image], ndarray]:
        images: Union[List[Image], ndarray] = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=28,
            width=width,
            height=height,
            guidance_scale=7.0,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
        ).images

        return images


class LoadUpscaler(CustomNode):
    """
    Loads an image upscaler model
    """

    display_name = {
        Language.ENGLISH: "Load Upscaler",
    }

    category = Category.UPSCALER

    description = {
        Language.ENGLISH: "Loads an image upscaler model",
    }

    @staticmethod
    def update_interface(_inputs: dict[str, Any] = None) -> NodeInterface:
        interface = {
            "inputs": {"model_path": StringInput()},
            "outputs": {"model": ModelConstraint()},
        }

        return interface

    def __call__(self, model_path: str, device: str) -> Dict[str, Any]:
        state_dict = load_models.load_state_dict_from_file(model_path, device)

        # Do we load using spandrel? or
        model = ModelLoader().load_from_state_dict(state_dict)
        model.to(torch.bfloat16)

        return {"model": model}


class UpscaleImage(CustomNode):
    """
    Takes an image and an image upscaler model, outputs an upscaled image
    """

    display_name = {
        Language.ENGLISH: "Upscale Image",
    }

    category = Category.UPSCALER

    description = {
        Language.ENGLISH: "Upscales an image using an image upscaler model",
    }

    @staticmethod
    def update_interface(inputs: dict[str, Any]) -> NodeInterface:
        interface = {
            "inputs": {
                "image_url": StringInput(),
                "model": ModelConstraint(),
            },
            "outputs": {"image": ImageOutputType},
        }

        return interface

    def __call__(
        self,
        model,
        image_path: str,
        *,
        output_keys: dict = {},
        device: TorchDevice = None,
    ):
        # file_path = get_input_file(image_path)
        with open(image_path, "rb") as f:
            try:
                img = Image()
                img.frombytes(f.read())
                tensor = pil_to_tensor(img)

                with torch.no_grad():
                    upscaled_image = model(tensor)
            except Exception as e:
                print(f"Upscale error: {e}")

        return {
            "image": tensor_to_pil(upscaled_image),
        }


class LoadLoRA(CustomNode):
    """
    Loads a LoRA model
    """

    display_name = {
        Language.ENGLISH: "Load LoRA",
    }

    category = Category.LOADER

    description = {
        Language.ENGLISH: "Loads a LoRA model onto the specified pipe",
    }

    @staticmethod
    def update_interface(inputs: dict[str, Any] = None) -> NodeInterface:
        interface = {
            "inputs": {
                "pipe": StableDiffusionPipeline,
                "model_path": StringInput(),
            },
            "outputs": {"pipe": StableDiffusionPipeline},
        }

        return interface

    def __call__(
        self, pipe: StableDiffusionPipeline, model_path: str
    ) -> Dict[str, Any]:
        pipe.load_lora_weights(
            model_path,
            weight_name=model_path,  # Using the model path as the weight name
        )

        return {"pipe": pipe}
