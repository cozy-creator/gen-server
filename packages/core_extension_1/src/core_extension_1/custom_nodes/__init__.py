import sys
from typing import TypedDict
from gen_server.utils import load_models
from gen_server.types import ModelWrapper, ArchDefinition, TorchDevice
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel


# TO DO: 'device' should somehow be marked as an internal-only parameter
# reserved just for the executor to have tighter control. It should NOT
# be a regular input. We do not want the end-user to worry about the device
# they are running on.
class LoadCheckpoint():
    def determine_output(self, file_path: str) -> dict[str, ArchDefinition]:
        return load_models.detect_all(file_path)
    
    def __call__(self, file_path: str, device: TorchDevice = None) -> dict[str, ModelWrapper]:
        return load_models.from_file(file_path, device)


class BuildPipe():
    def determine_output():
        pass
    
    # TO DO: we also need to specify the input / output space for each model
    # in addition to the class
    # custom nodes shouldn't have to spend much time validating / sanitizing their inputs
    # that should be the executor's job. The custom nodes should be delcarative
    def __call__(vae: AutoencoderKL, text_encoder: CLIPTextModel, unet: UNet2DConditionModel, device: TorchDevice = None):
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        # tokenizer_2 = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
        scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    
        pipe = StableDiffusionPipeline(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        if "xformers" in sys.modules:
            pipe.enable_xformers_memory_efficient_attention()
        if "accelerate" in sys.modules:
            pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        pipe.to(device)
        
        return pipe