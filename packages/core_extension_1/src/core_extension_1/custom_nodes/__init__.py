import sys
from typing import Any, Union, List, Optional, Dict, Tuple
from pathlib import Path

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
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    EDMDPMSolverMultistepScheduler,
    EDMEulerScheduler,
    StableDiffusionXLInpaintPipeline
)
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    T5TokenizerFast,
    CLIPTextModelWithProjection,
    T5EncoderModel,
)

from core_extension_1.widgets import TextInput, StringInput, EnumInput, IntInput, FloatInput, BooleanInput
import json
import torch
import os
from PIL.Image import Image
from numpy import ndarray
from spandrel import ModelLoader

from image_utils.custom_nodes import pil_to_tensor, tensor_to_pil
import logging
from huggingface_hub import hf_hub_download
import requests
from tqdm import tqdm
from gen_server.config import get_config

# Masking
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops
from core_extension_1.common.utils import resize_and_crop, load_image_from_url, save_tensor_as_image
import numpy as np
import scipy.ndimage
from diffusers.utils import load_image


# Configure the logging
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)

config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scheduler_config.json"
)

config_path_play = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.json"
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
    ux_node = "../ux_node.json"
    ux_node2 = "../ux_node.tsx"

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
                    loaded_component = class_.from_pretrained(repo_id, subfolder=component, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
                    loaded_components[component] = loaded_component
                else:
                    loaded_components[component] = None

            # print(loaded_components)

            return loaded_components

        except Exception as e:
            logging.error(f"Error loading components: {e}")
            return {}


class LoadCivitai(CustomNode):
    """
    Node to load model components from Civitai based on user selection.
    """

    display_name = {
        "ENGLISH": "Load Components",
        "CHINESE": "加载组件",
    }

    category = "LOADER"

    description = {
        "ENGLISH": "Loads selected components from a repository and returns a dictionary of model-classes",
        "CHINESE": "从存储库中加载所选组件，并返回模型类的字典",
    }

    @staticmethod
    def update_interface(inputs: Dict[str, Any]) -> Dict[str, Any]:
        interface = {
            "inputs": {
                "model_name": "text",
                "version_name": "text"
            },
            "outputs": {
                "model_file": "file"
            },
        }
        if inputs:
            model_name = inputs.get("model_name")
            if model_name:
                model_id = LoadCivitai.search_model_by_name(model_name)
                if model_id:
                    interface["outputs"]["model_file"] = "file"
                else:
                    interface["outputs"]["error"] = "text"

        return interface

    @staticmethod
    def search_model_by_name(model_name: str):
        try:
            # Search for models by name
            search_response = requests.get(f"https://api.civitai.com/v1/models?query={model_name.lower()}")
            search_response.raise_for_status()
            search_results = search_response.json()

            if not search_results or search_results['items'] == []:
                logging.error(f"No models found with the name: {model_name}")
                return None

            # Assuming the first result is the desired model
            # Should we use id instead of name to get the exact model? Will that be too complex for users?
            model_id = search_results['items'][0]['id']
            return model_id

        except requests.RequestException as e:
            logging.error(f"Error searching for models: {e}")
            return None

    @staticmethod
    def download_model(repo_id: int, version_name: str = None):
        try:
            response = requests.get(f"https://civitai.com/api/v1/models/{repo_id}")
            response.raise_for_status()
            model_details = response.json()

            # Find the correct file version
            file_version = next((v for v in model_details['modelVersions'] if version_name in v['name']), model_details['modelVersions'][0])
            file_info = file_version['files'][0]
            file_size_kb = file_info['sizeKB']
            file_name = file_info['name']
            download_url = f"{file_info['downloadUrl']}?token={os.getenv('civitaiToken')}"

            model_path = os.path.join(get_config().workspace_dir, "models", file_name)

            # Check if the file already exists and its size
            file_exists = os.path.exists(model_path)
            existing_file_size = os.path.getsize(model_path) if file_exists else 0

            if file_exists and existing_file_size >= file_size_kb * 1024:
                logging.info(f"Model '{file_name}' is already fully downloaded.")
                # loaded component
                return model_path

            # Convert file size from KB to bytes
            total_size = int(file_size_kb * 1024)

            # Download model with progress bar and resume capability
            headers = {"Range": f"bytes={existing_file_size}-"}
            response = requests.get(download_url, headers=headers, stream=True)
            response.raise_for_status()

            mode = 'ab' if file_exists else 'wb'
            with open(model_path, mode) as f:
                with tqdm(total=total_size, initial=existing_file_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            return model_path

        except requests.RequestException as e:
            logging.error(f"Request error: {e}")
        except Exception as e:
            logging.error(f"Error downloading model: {e}")

    def __call__(self, model_name: str, version_name: str = None, device: Optional[TorchDevice] = None) -> dict[str, Architecture]:
        model_id = self.search_model_by_name(model_name)
        if not model_id:
            return {"error": f"No models found with the name: {model_name}"}
        
        model_path = self.download_model(model_id, version_name)
        if not model_path:
            return {"error": "Failed to download model."}
        

        return load_models.from_file(model_path, device)




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
        model_type: str = None
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
            if model_type == "playground":
                tokenizer = CLIPTokenizer.from_pretrained("playgroundai/playground-v2.5-1024px-aesthetic", subfolder="tokenizer")
                scheduler = EDMDPMSolverMultistepScheduler.from_pretrained("playgroundai/playground-v2.5-1024px-aesthetic", subfolder="scheduler")
                tokenizer_2 = CLIPTokenizer.from_pretrained(
                    "playgroundai/playground-v2.5-1024px-aesthetic", subfolder="tokenizer_2"
                )
            else:
                tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer")
                scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
                tokenizer_2 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")

            pipe = StableDiffusionXLPipeline(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                unet=unet,
                scheduler=scheduler,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
            ).to(torch.float16)
        else:
            # EulerDiscreteScheduler
            # runwayml/stable-diffusion-v1-5
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

        print("Got Here")

        # if "xformers" in sys.modules:
        #     pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        if "accelerate" in sys.modules:
            pipe.enable_model_cpu_offload()
        # pipe.enable_vae_tiling()
        # pipe.to(device)

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
            # negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            # width=width,
            # height=height,
            guidance_scale=guidance_scale,
            # num_images_per_prompt=num_images,
            # generator=generator,
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
    

class GenerateMaskInpainting(CustomNode):
    """
    Generate mask based on the text prompt. This mask will be used for inpainting
    """

    display_name = {
        Language.ENGLISH: "Generate Mask",
    }

    category = Category.MASK

    description = {
        Language.ENGLISH: "Generate Mask on specific portion of an image based on the text prompt",
    }

    @staticmethod
    def update_interface(inputs: Dict[str, Any] = None) -> NodeInterface:
        interface = {
            "inputs": {
                "mask_prompt": TextInput(),
                "feather": IntInput(),
                "image": StringInput(),
            },
            "outputs": {"image": ImageOutputType},
        }

        return interface
    
    def __call__(
        self,
        image: Union[Path, str],
        mask_prompt: str,
        feather: int,
    ) -> Tuple[Image.Image, Tuple[int, int]]:
        # Load models
        sam_predictor = self.get_sam()
        grounding_dino_model = self.get_groundingdino()

        # Load and process image
        try:
            image = load_image_from_url(image) if isinstance(image, str) else load_image(image)
        except:
            image = load_image(image)

        image, img_size = resize_and_crop(image)
        
        image_np = np.array(image)
        sam_predictor.set_image(image_np)

        # Detect objects using GroundingDINO
        boxes, logits, phrases = self.detect_objects(image, mask_prompt, grounding_dino_model)

        if len(boxes) > 0:
            combined_mask = np.zeros(image_np.shape[:2], dtype=bool)
            
            for i, box in enumerate(boxes):
                # Generate mask using SAM
                input_box = box.cpu().numpy()
                masks, _, _ = sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                
                combined_mask = np.logical_or(combined_mask, masks[0])

            mask_image_pil = Image.fromarray(combined_mask.astype(np.uint8) * 255)
            mask_image_pil.save("wmask.png")
            
            # Feather the mask
            combined_mask = self.feather_mask(combined_mask, iterations=feather)
            
            mask_image_pil = Image.fromarray(combined_mask.astype(np.uint8) * 255)
            mask_image_pil.save("wmfeathermask.png")

            return mask_image_pil, img_size
        else:
            print(f"No objects matching '{mask_prompt}' found in the image.")
            return None, img_size
    
    def get_sam(self) -> SamPredictor:
        sam_checkpoint = "models/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        return SamPredictor(sam)
    
    def get_groundingdino(self) -> torch.nn.Module:
        config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        checkpoint_file = "models/groundingdino_swint_ogc.pth"
        model = load_model(config_file, checkpoint_file)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        return model

    def transform_image(self, image: Image.Image) -> torch.Tensor:
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(image, None)
        return image_transformed
    
    def detect_objects(self, image: Image.Image, text_prompt: str, model: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        image_transformed = self.transform_image(image)
        boxes, logits, phrases = predict(
            model=model, 
            image=image_transformed, 
            caption=text_prompt, 
            box_threshold=0.3, 
            text_threshold=0.25
        )

        W, H = image.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([W, H, W, H])
        return boxes, logits, phrases
    
    def feather_mask(self, mask: np.ndarray, iterations: int = 5) -> np.ndarray:
        """
        Feather the mask to increase its size and soften the edges.
        """
        mask = mask.astype(np.float32)
        for _ in range(iterations):
            mask = scipy.ndimage.gaussian_filter(mask, sigma=1)
            mask[mask > 0] = 1  # Threshold to keep mask binary
        return mask
    

class InpaintImage(CustomNode):
    """
    Regenerate an image based on the mask and text prompt
    """

    display_name = {
        Language.ENGLISH: "Regenerate Image (Inpainting)",
    }

    category = Category.INPAINTING

    description = {
        Language.ENGLISH: "Regenerate an image based on the mask and text prompt",
    }

    @staticmethod
    def update_interface(inputs: dict[str, Any] = None) -> NodeInterface:
        interface = {
            "inputs": {
                "image": Image,
                "mask": ImageOutputType,
                "text_prompt": TextInput(),
                "strength": FloatInput(),
                "save": BooleanInput()
            },
            "outputs": {"image": ImageOutputType},
        }

        return interface
    
    def __call__(
        image,
        mask,
        text_prompt,
        strength,
        save=False
    ):
        inpainter = StableDiffusionXLInpaintPipeline.from_pretrained("RunDiffusion/Juggernaut-XL-v9", variant="fp16", torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
    
        mask, img_size = mask

        # Perform inpainting
        with torch.no_grad():
            output = inpainter(
                prompt=text_prompt,
                negative_prompt="bad quality, bad teeth, worst quality, sweat",
                image=image,
                mask_image=mask,
                strength=strength,
                guidance_scale=7.5,
                num_inference_steps=50,
                width=img_size[0],
                height=img_size[1],
                output_type="latent",
            )

            print("Done Muahahaha!!!")
        
        if save:
            save_tensor_as_image(output.images, inpainter.vae, "inpainting.jpg")

        return output.images, inpainter.vae  # Return both the latent tensor and the VAE

    
    

