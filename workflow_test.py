import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline
from diffusers.models import ControlNetModel
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import numpy as np
from controlnet_aux import OpenposeDetector, MLSDdetector, MidasDetector    # pip install controlnet_aux (note you'd have to reinstall torch again 🥲. Use this command instead to avoid conflict `pip install torch==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html`)
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils import load_image
from briarmbg_arch.briarmbg import BriaRMBG    # This should be synced with the bg remover Rahman implemented
import cv2
from depth_anything.depth_anything_v2.dpt import DepthAnythingV2


# Note that the ip_embeddings has been implemented here due to its slow inference

def resize_and_crop(image_path, target_size):
    with Image.open(image_path) as img:
        # Calculate aspect ratios
        aspect_ratio_img = img.width / img.height
        aspect_ratio_target = target_size[0] / target_size[1]
        
        if aspect_ratio_img > aspect_ratio_target:
            # Image is wider than target, resize based on height
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio_img)
        else:
            # Image is taller than target, resize based on width
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio_img)
        
        # Resize the image
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Calculate crop box
        left = (new_width - target_size[0]) / 2
        top = (new_height - target_size[1]) / 2
        right = (new_width + target_size[0]) / 2
        bottom = (new_height + target_size[1]) / 2
        
        # Crop the image
        img_cropped = img_resized.crop((left, top, right, bottom))
        
        return img_cropped
    

def pil_to_cv2(pil_image):
    # Convert PIL image to RGB (OpenCV uses BGR)
    rgb_image = pil_image.convert('RGB')
    
    # Convert to numpy array
    numpy_image = np.array(rgb_image)
    
    # Convert RGB to BGR
    bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    
    return bgr_image



# Usage
# image = resize_and_crop('64da8c1728d7e544d48a0cd43beb4087.jpg', (1024, 1024))
# image.save('resized_and_cropped_image.jpg')



# # from depth_anything_v2.dpt


# def load_image(image_path):
#     return Image.open(image_path).convert("RGB")

# def resize_image(image, size=(1024, 1024)):
#     return image.resize(size, Image.LANCZOS)

def extract_openpose(image):
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    openpose_image = openpose(image)
    return openpose_image

def extract_depth_map(image):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'models/depth_anything_v2_vitl.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    raw_img = pil_to_cv2(image)

    # raw_img = cv2.imread(image)
    depth = model.infer_image(raw_img) # HxW raw depth map in numpy

    # Normalize the depth map
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8
    depth_norm = np.uint8(depth_norm)

    # Apply a colormap
    depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    # Save the depth map
    cv2.imwrite('depth_map.png', depth_norm)

    # # Save the colored depth map
    # cv2.imwrite('depth_map_color.png', depth_colormap)

    # Convert BGR to RGB
    depth_colormap_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    depth_colormap_pil = Image.fromarray(depth_colormap_rgb)

    return depth_colormap_pil


# def extract_depth_map(image):
#     depth_estimator = MidasDetector.from_pretrained("lllyasviel/ControlNet")
#     depth_map = depth_estimator(image)
#     return depth_map


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
        return mask
    return None


# def select_face_area(image):
#     from facenet_pytorch import MTCNN
#     from PIL import Image, ImageDraw
#     import numpy as np

#     mtcnn = MTCNN(keep_all=True)
#     boxes, _, landmarks = mtcnn.detect(image, landmarks=True)

#     if boxes is not None and len(boxes) > 0:
#         box = boxes[0]
#         x1, y1, x2, y2 = [int(b) for b in box]
        
#         # Create a black mask
#         mask = Image.new("L", image.size, 0)
#         draw = ImageDraw.Draw(mask)
        
#         # Get facial landmarks
#         face_landmarks = landmarks[0]
        
#         # Create a polygon from the landmarks
#         polygon = []
#         for landmark in face_landmarks:
#             polygon.extend(landmark)
        
#         # Draw the face polygon
#         draw.polygon(polygon, fill=255)
        
#         # Smooth the edges
#         mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
        
#         # Resize the mask to 1024x1024
#         return mask.resize((1024, 1024), Image.LANCZOS)

#     return None




def remove_background(image):
    import torchvision.transforms as transforms
    # Convert the PIL Image to a PyTorch Tensor
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Load the BriaRMBG model
    pipe = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

    # Run the model
    with torch.no_grad():
        output = pipe(image_tensor)
        if isinstance(output, list):
            output_tensor = output[0][0]  # Assuming we need the first element of the list
        elif isinstance(output, tuple):
            output_tensor = output[0][0]  # Extract the first element if it's a tuple
        else:
            output_tensor = output

    # Convert the output Tensor back to a PIL Image
    mask = transforms.ToPILImage()(output_tensor.squeeze(0))  # Remove batch dimension

    # Convert the original image and mask to RGBA
    original_image_rgba = image.convert("RGBA")
    mask_rgba = mask.convert("L").resize(original_image_rgba.size)
    
    # Composite the original image with the mask
    background = Image.new("RGBA", original_image_rgba.size, (0, 0, 0, 0))
    composite_image = Image.composite(original_image_rgba, background, mask_rgba)

    return composite_image

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

    sd_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained("RunDiffusion/Juggernaut-XL-v9", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16")

    sd_main_pipeline = StableDiffusionXLPipeline.from_pretrained("Lykon/dreamshaper-xl-v2-turbo", torch_dtype=torch.float16, variant="fp16")

    inpaint_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained("Lykon/dreamshaper-xl-v2-turbo", variant="fp16", torch_dtype=torch.float16)

    sd_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    sd_pipeline.set_ip_adapter_scale(0.6)

    inpaint_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    inpaint_pipeline.set_ip_adapter_scale(0.1)

    # stabilityai/stable-diffusion-xl-base-1.0
    # diffusers/stable-diffusion-xl-1.0-inpainting-0.1
    # RunDiffusion/Juggernaut-XL-v9
    # Lykon/dreamshaper-xl-v2-turbo

    
    
    
    # Load images
    # image1 = load_image("64da8c1728d7e544d48a0cd43beb4087.jpg")
    # image2 = load_image("64da8c1728d7e544d48a0cd43beb4087.jpg")
    image1 = resize_and_crop("64da8c1728d7e544d48a0cd43beb4087.jpg", (1024, 1024))
    image2 = resize_and_crop("64da8c1728d7e544d48a0cd43beb4087.jpg", (1024, 1024))

    ip_adapter_images = resize_and_crop("cypress2.jpg", (1024, 1024)) # Get embeddings instead



    # prompt = "girl standing and wearing pink top and black jeans, plain background, photorealism, realistic"
    # negative_prompt = "blurry"
    # prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
    #     ip_adapter_images, prompt=prompt, negative_prompt=negative_prompt
    # )


    # image = sd_pipeline(
    #     prompt_embeds=prompt_embeds,
    #     negative_prompt_embeds=negative_prompt_embeds,
    #     num_inference_steps=30,
    #     guidance_scale=6.0,
    # ).images[0]

    # image.save("image.webp", lossless=True, quality=100)


    # 64da8c1728d7e544d48a0cd43beb4087
    
    # Extract features
    openpose_image = extract_openpose(image1)
    depth_map = extract_depth_map(image2)
    # ip_embeddings = get_ip_adapter_embedding(ip_adapter_images)


    openpose_image.save("wopenpose.png")
    depth_map.save("wdepthmap.png")
    

    # Generate text embedding
    text_prompt = "girl standing and wearing pink top and black jeans, plain background, photorealism, realistic"
    

    print("Done the prerequisites!")

    sd_pipeline.enable_model_cpu_offload()
    inpaint_pipeline.enable_model_cpu_offload()
    sd_main_pipeline.enable_model_cpu_offload()

    print("Using model_cpu_offload")

    # Generate embeddings TODO
    # image_embeds = sd_pipeline.prepare_ip_adapter_image_embeds(
    #     ip_adapter_image=ip_adapter_images,
    #     ip_adapter_image_embeds=None,
    #     device="cuda",
    #     num_images_per_prompt=1,
    #     do_classifier_free_guidance=True,
    # )

    # torch.save(image_embeds, "image_embeds.ipadpt")
    
    # First image generation
    generated_image = sd_pipeline(
        prompt=text_prompt,
        negative_prompt="bad, red tint",
        image=[openpose_image, depth_map],
        ip_adapter_image=ip_adapter_images,
        controlnet_conditioning_scale=0.7,
        num_inference_steps=25,
        height=1024,
        width=1024,
        # guidance_scale=5
    ).images[0]

    print("Done generating")

    generated_image.save("workflow.png")
    
    # Select face area and create mask
    face_mask = select_face_area(generated_image)

    face_mask.save("wfacemask.png")
    
    # Image regeneration with inpainting
    if face_mask:
        text_prompt = "smiling, masterpiece, best quality, 8k"
        inpainted_image = inpaint_pipeline(
            prompt=text_prompt,
            negative_prompt="bad eye, malformed",
            image=generated_image,
            mask_image=face_mask,
            ip_adapter_image=ip_adapter_images,
            num_inference_steps=25,
            strength=0.4
        ).images[0]
    else:
        inpainted_image = generated_image

    inpainted_image.save("workflow2.png")

    # del sd_pipeline, inpaint_pipeline

    
    # Remove background
    foreground_image = remove_background(inpainted_image)

    foreground_image.save("wforeground.png")
    
    # Generate new background
    background_prompt = "beautiful landscape, snow, mountains, glaciers, vivid colors"
    background_image = sd_main_pipeline(prompt=background_prompt, num_inference_steps=25).images[0]

    background_image.save("wbackground.png")
    
    # Composite images
    final_image = composite_images(foreground_image, background_image)
    
    # Save the result
    final_image.save("wfinal_output.png")

    del sd_pipeline, sd_main_pipeline, inpaint_pipeline

if __name__ == "__main__":
    main()

    