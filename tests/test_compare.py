import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import StableDiffusionXLPipeline

def compare_text_encoders(model1, model2):

    encoder1_params = dict(model1.text_encoder_2.named_parameters())
    encoder2_params = dict(model2.text_encoder_2.named_parameters())
    
    if encoder1_params.keys() != encoder2_params.keys():
        return False
    
    mismatched = []

    for name in encoder1_params:
        if not torch.equal(encoder1_params[name], encoder2_params[name]):
            print(f"Mismatch found in parameter: {name}")
            mismatched.append(name)
    
    return mismatched


# pipe = CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
# pipe2 = CLIPTextModelWithProjection.from_pretrained("playgroundai/playground-v2.5-1024px-aesthetic")


# pipe = StableDiffusionXLPipeline.from_pretrained("playgroundai/playground-v2.5-1024px-aesthetic", variant="fp16", torch_dtype=torch.float16)
pipe2 = StableDiffusionXLPipeline.from_pretrained("RunDiffusion/Juggernaut-XL-v9", variant="fp16", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_single_file("D:/models/sd_xl_base_1.0.safetensors", torch_dtype=torch.float16)
# pipe2 = StableDiffusionXLPipeline.from_single_file("D:/models/breakdomainxl_V06d.safetensors", torch_dtype=torch.float16)


mismatched = compare_text_encoders(pipe, pipe2)

if mismatched:
    print("The text encoders are different.")
    print(mismatched)
else:
    print("The text encoders are identical.")