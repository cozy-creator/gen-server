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
import logging
from huggingface_hub import hf_hub_download


# Configure the logging
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)

config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scheduler_config.json"
)


def components_from_model_index(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        data = json.load(file)
    return {key: value for key, value in data.items() if isinstance(value, list)}


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
        self,
        file_path: str,
        *,
        output_keys: dict = {},
        device: Optional[TorchDevice] = None,
    ) -> dict[str, Architecture]:
        return load_models.from_file(file_path, device)


class LoadComponents(CustomNode):
    """
    Node to load model components from a repository based on user selection.
    """

    display_name = {
        Language.ENGLISH: "Load Components",
        Language.CHINESE: "加载组件",
    }

    category = Category.LOADER

    description = {
        Language.ENGLISH: "Loads selected components from a repository and returns a dictionary of model-classes",
        Language.CHINESE: "从存储库中加载所选组件，并返回模型类的字典",
    }

    @staticmethod
    def update_interface(inputs: Dict[str, Any]) -> NodeInterface:
        interface = {
            "inputs": {
                "repo_id": EnumInput(options=[]),
                "components": EnumInput(options=[]),
            },
            "outputs": {},
        }
        if inputs:
            repo_id = inputs.get("repo_id")
            if repo_id:
                try:
                    path = hf_hub_download(repo_id, "model_index.json")
                    components = components_from_model_index(path)
                    interface["inputs"]["components"] = EnumInput(
                        options=list(components.keys())
                    )
                    interface["outputs"] = {
                        component: str for component in components.keys()
                    }
                except Exception as e:
                    logging.error(f"Error loading components: {e}")

        return interface

    def __call__(self, repo_id: str, components: List[str]) -> Dict[str, Any]:
        try:
            path = hf_hub_download(repo_id, "model_index.json")
            with open(path, "r") as file:
                data = json.load(file)

            loaded_components = {}
            if "_class_name" in data:
                loaded_components["_class_name"] = data["_class_name"]

            for component in data.keys():
                if component.startswith("_") or component == "_class_name":
                    # print(component)
                    continue

                if component in components:
                    module_name, class_name = data[component]
                    module = __import__(module_name, fromlist=[class_name])
                    class_ = getattr(module, class_name)
                    loaded_component = class_.from_pretrained(
                        repo_id, subfolder=component, torch_dtype=torch.float16
                    )
                    loaded_components[component] = loaded_component
                else:
                    loaded_components[component] = None

            # print(loaded_components)

            return loaded_components

        except Exception as e:
            logging.error(f"Error loading components: {e}")
            return {}


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
        loaded_components: Optional[Dict[str, Any]] = None,
        unet: Union[UNet2DConditionModel, SD3Transformer2DModel] = None,
        vae: AutoencoderKL = None,
        text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection] = None,
        text_encoder_2: Optional[CLIPTextModelWithProjection] = None,
        text_encoder_3: Optional[T5EncoderModel] = None,
        device: Optional[TorchDevice] = None,
    ) -> Union[StableDiffusion3Pipeline, StableDiffusionXLPipeline]:
        if loaded_components:
            class_name = loaded_components.pop("_class_name", None)
            if class_name:
                # Dynamically fetch the required components from loaded_components
                component_kwargs = {k: v for k, v in loaded_components.items()}

                # Dynamically fetch the class
                module = __import__("diffusers", fromlist=[class_name])
                class_ = getattr(module, class_name)
                # print(class_)
                # Instantiate the class with the provided components
                pipe = class_(**component_kwargs)
                pipe.to("cuda")
                return pipe

        # Default behavior when loaded_components is not provided
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
        num_images: int = 1,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Union[List[Image], ndarray]:
        images: Union[List[Image], ndarray] = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        ).images

        return images


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
        image: torch.Tensor,
        *,
        output_keys: dict = {},
        device: TorchDevice = None,
    ):
        try:
            with torch.no_grad():
                upscaled_image = model(image)
        except Exception as e:
            print(f"Upscale error: {e}")

        return {"image": upscaled_image}


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
                "model_path": EnumInput(),
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
