import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline
from diffusers.models import ControlNetModel
from PIL import Image
import numpy as np
from controlnet_aux import OpenposeDetector, MLSDdetector, MidasDetector  # pip install controlnet_aux (note you'd have to reinstall torch again 🥲. Use this command instead to avoid conflict `pip install torch==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html`)
from transformers import CLIPTextModel, CLIPTokenizer
# from ip_adapter import IPAdapterPlus, IPAdapterPlusXL


def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def resize_image(image, size=(1024, 1024)):
    return image.resize(size, Image.LANCZOS)

def extract_openpose(image):
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    openpose_image = openpose(image)
    return resize_image(openpose_image)

def extract_depth_map(image):
    depth_estimator = MidasDetector.from_pretrained("lllyasviel/ControlNet")
    depth_map = depth_estimator(image)
    return resize_image(depth_map)

def select_face_area(image):
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=True)
    boxes, _ = mtcnn.detect(image)
    if boxes is not None and len(boxes) > 0:
        box = boxes[0]
        x1, y1, x2, y2 = [int(b) for b in box]
        # Create a black mask
        mask = Image.new("L", image.size, 0)
        # Draw a white rectangle on the face area
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x1, y1, x2, y2], fill=255)
        # Resize the mask to 1024x1024
        return resize_image(mask, size=(1024, 1024))
    return None


# def get_ip_adapter_embedding(images):
#     ip_adapter = IPAdapterPlus(sd_version='2.1', device='cuda')
#     ip_embeddings = ip_adapter.get_image_embeds(images)
#     return ip_embeddings


def remove_background(image):
    # Placeholder
    return image  # For now, just return the original image

def composite_images(foreground, background):
    return Image.alpha_composite(background.convert("RGBA"), foreground.convert("RGBA"))

def main():
    # Load models
    openpose_controlnet = ControlNetModel.from_pretrained("xinsir/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16)
    depthmap_controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, variant="fp16")

    # diffusers/controlnet-depth-sdxl-1.0
    # lllyasviel/control_v11f1p_sd15_depth

    # thibaud/controlnet-openpose-sdxl-1.0

    controlnet = [openpose_controlnet, depthmap_controlnet]

    sd_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16")

    inpaint_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", variant="fp16", torch_dtype=torch.float16)
    
    
    # Load images
    image1 = load_image("pexels-niko-twisty-4048182.jpg")
    image2 = load_image("pexels-niko-twisty-4048182.jpg")
    ip_adapter_images = [load_image("output.png")]
    
    # Extract features
    openpose_image = extract_openpose(image1)
    depth_map = extract_depth_map(image2)
    # ip_embeddings = get_ip_adapter_embedding(ip_adapter_images)

    openpose_image.save("wopenpose.png")
    depth_map.save("wdepthmap.png")
    
    # Generate text embedding
    text_prompt = "A beautiful lady running in the garden"

    sd_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    sd_pipeline.set_ip_adapter_scale(0.1)

    sd_pipeline.enable_model_cpu_offload()
    
    # First image generation
    generated_image = sd_pipeline(
        prompt=text_prompt,
        image=[openpose_image, depth_map],
        ip_adapter_image=ip_adapter_images,
        controlnet_conditioning_scale=0.1,
        num_inference_steps=25
    ).images[0]

    generated_image.save("workflow.png")
    
    # Select face area and create mask
    face_mask = select_face_area(generated_image)

    face_mask.save("wfacemask.png")

    inpaint_pipeline.enable_model_cpu_offload()
    
    # Image regeneration with inpainting
    if face_mask:
        inpainted_image = inpaint_pipeline(
            prompt=text_prompt,
            image=generated_image,
            mask_image=face_mask,
            num_inference_steps=25
        ).images[0]
    else:
        inpainted_image = generated_image

    inpainted_image.save("workflow2.png")

    
    # # Remove background
    # foreground_image = remove_background(inpainted_image)
    
    # # Generate new background
    # background_prompt = "A beautiful garden landscape"
    # background_image = sd_pipeline(prompt=background_prompt, num_inference_steps=30).images[0]
    
    # # Composite images
    # final_image = composite_images(foreground_image, background_image)
    
    # # Save the result
    # final_image.save("final_output.png")

if __name__ == "__main__":
    main()

    