from diffusers.utils import load_image
import cv2
from PIL import Image
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import torch
import time
from typing import Union
from paths import get_model_path, check_model_in_path


MODEL_PATH = get_model_path('models')


class LoadControlNet:
    def __init__(self):
        pass

    @classmethod
    def run(self, model_id: str) -> ControlNetModel:
        model_path = check_model_in_path(model_id, MODEL_PATH)

        if model_path is None:
            controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
        else:
            controlnet = ControlNetModel.from_single_file(model_path, torch_dtype=torch.float16)

        return controlnet

    
class LoadPipeline:
    def __init__(self):
        pass

    @classmethod
    def run(self, model_id: str, controlnet: ControlNetModel, torch_dtype: Union[str, torch.dtype]) -> StableDiffusionControlNetPipeline:
        model_path = check_model_in_path(model_id, MODEL_PATH)

        if model_path is None:

            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_id, controlnet=controlnet, torch_dtype=torch_dtype
            )
        else:
            pipe = StableDiffusionControlNetPipeline.from_single_file(
                model_path, controlnet=controlnet, torch_dtype=torch_dtype
            )

        pipe = pipe.to("cuda")
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # pipe.enable_model_cpu_offload()
        
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_vae_tiling()
        return pipe
    

class LoadImage:
    def __init__(self):
        pass

    @classmethod
    def run(self, image_path: str) -> Image.Image:
        image = load_image(image_path)
        return image


class CannyImage:
    def __init__(self):
        pass

    @classmethod
    def run(self, image: Image, low_threshold: int, high_threshold: int) -> Image.Image:
        image = load_image(image)
        image = np.array(image)
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        return canny_image



class RunPipeline:
    def __init__(self, pipe, prompt: str, negative_prompt: str, num_inference_steps: int, generator: torch.Generator, image: bytes = None):
        self.pipe = pipe
        self.prompt = prompt
        self.canny_image = image
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.generator = generator

    def run(self):
        output = self.pipe(
            self.prompt,
            image=self.canny_image,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            generator=self.generator,
        )
        return output
    

class SaveImage:
    def __init__(self):
        pass

    @classmethod
    def run(self, image: bytes) -> bytes:
        for i in range(len(image)):
            image[i].save(f"files/image_output_{i}.png")
        return image
    

