from diffusers.utils import load_image
import cv2
from PIL import Image
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import torch
import time
from typing import Union, List
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
    def run(self, image: bytes, low_threshold: int, high_threshold: int) -> Image.Image:
        image = load_image(image)
        image = np.array(image)
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        return canny_image



class RunPipeline:
    def __init__(self, pipe, prompt: str, 
                 negative_prompt: str, 
                 num_inference_steps: int, 
                 generator: torch.Generator, 
                #  height: int,
                #  width: int,
                 image: bytes = None,
                 latent_preview: bool = False,
                 ) -> Union[bytes, List[bytes]]:
        
        self.pipe = pipe
        self.prompt = prompt
        self.canny_image = image
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.generator = generator
        self.latent_preview = latent_preview
        # self.height = height
        # self.width = width
        self.latent_images = []
        self.vae = self.pipe.vae

    def run(self):
        if self.latent_preview:
            output = self.pipe(self.prompt, image=self.canny_image, negative_prompt=self.negative_prompt, 
                               num_inference_steps=self.num_inference_steps, callback_on_step_end=self.decode_tensors, 
                               callback_on_step_end_tensor_inputs=["latents"])
            return output
        else:
            output = self.pipe(
                self.prompt,
                image=self.canny_image,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.num_inference_steps,
                generator=self.generator,
            )
            return output
        
    def latents_callback(self, i, t, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(1, 2, 0).numpy()
        pil_image = self.pipe.numpy_to_pil(image)
        pil_image[-1].save(f"./latents/{t}.png")
        # self.latent_images.extend(pil_image)


    def latents_to_rgb(self, latents):
        weights = (
            (60, -60, 25, -70),
            (60,  -5, 15, -50),
            (60,  10, -5, -35)
        )

        weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
        biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
        rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
        image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
        image_array = image_array.transpose(1, 2, 0)

        return Image.fromarray(image_array)

    def decode_tensors(self, pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        
        image = self.latents_to_rgb(latents)
        image.save(f"./latents/{step}.png")

        return callback_kwargs
    

class SaveImage:
    def __init__(self):
        pass

    @classmethod
    def run(self, image: bytes) -> bytes:
        for i in range(len(image)):
            image[i].save(f"files/image_output_{i}.png")
        return image
    


