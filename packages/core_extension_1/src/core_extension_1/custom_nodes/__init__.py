import sys
from typing import Any, Union, List
from gen_server.utils import load_models
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
from gen_server.globals import MODEL_FILES
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusion3Pipeline,
    DDIMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    SD3Transformer2DModel,
    FlowMatchEulerDiscreteScheduler
)
from transformers import CLIPTokenizer, CLIPTextModel, T5TokenizerFast, CLIPTextModelWithProjection, T5EncoderModel

from core_extension_1.widgets import TextInput, StringInput, EnumInput
import json
import torch
import os
# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp


config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scheduler_config.json")

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
            "inputs": { "file_path": EnumInput(options=list(MODEL_FILES.keys())) }, 
            "outputs": {} 
        }
        if inputs:
            file_path = inputs.get("file_path", None)
            if file_path:
                interface.update({"outputs": load_models.detect_all(file_path)})

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
        text_encoder_2: CLIPTextModelWithProjection = None,
        text_encoder_3: T5EncoderModel = None,
        device: TorchDevice = None,
    ) -> StableDiffusion3Pipeline:
        
        if isinstance(unet, SD3Transformer2DModel) and text_encoder_3 is not None:
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            tokenizer_2 = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
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
            pass # SDXL
        else:
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
        self, pipe: StableDiffusionPipeline, prompt: str, negative_prompt: str = None
    ) -> ImageOutputType:
        images: ImageOutputType = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.0,
            # num_images_per_prompt=4,
        ).images

        return images
