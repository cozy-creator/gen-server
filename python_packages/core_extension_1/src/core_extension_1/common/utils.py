import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import aiohttp


sdxl_resolutions = {
    (1, 1): (1024, 1024),
    (4, 3): (1152, 864),
    (3, 4): (896, 1152),
    (3, 2): (1248, 832),
    (2, 3): (832, 1248),
    (16, 9): (1344, 768),
    (9, 16): (768, 1344)
}


def get_closest_aspect_ratio(img_aspect_ratio):
    closest_aspect_ratio = None
    min_diff = float('inf')
    
    for aspect_ratio in sdxl_resolutions.keys():
        diff = abs(img_aspect_ratio - (aspect_ratio[0] / aspect_ratio[1]))
        if diff < min_diff:
            min_diff = diff
            closest_aspect_ratio = aspect_ratio

    print(f"Closest Ratio: {closest_aspect_ratio}")
    
    return closest_aspect_ratio


def resize_and_crop(img):
    # Calculate image aspect ratio
    img_aspect_ratio = img.width / img.height
    print(img_aspect_ratio)
    
    # Get the closest aspect ratio from SDXL models
    closest_aspect_ratio = get_closest_aspect_ratio(img_aspect_ratio)
    target_size = sdxl_resolutions[closest_aspect_ratio]
    
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
    
    return img_cropped, target_size


async def load_image_from_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.read()
    img = Image.open(BytesIO(content))
    return img.convert("RGB")


def save_tensor_as_image(latents, vae, filename):

    # Type checking and upcasting
    needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
    if needs_upcasting:
        vae = vae.float()
        latents = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype)


    # Latent denormalization
    has_latents_mean = hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None
    has_latents_std = hasattr(vae.config, "latents_std") and vae.config.latents_std is not None
    if has_latents_mean and has_latents_std:
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / vae.config.scaling_factor + latents_mean
    else:
        latents = latents / vae.config.scaling_factor

    # VAE decoding
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]

    # Normalize the image tensor
    image = (image / 2 + 0.5).clamp(0, 1)
    
    # # Convert to CPU and then to numpy array
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()

    # Convert to uint8 and create a PIL Image
    image = (image * 255).round().astype("uint8")[0]
    image = Image.fromarray(image)

    # Use torchvision to convert to PIL Image (stays on GPU as long as possible)
    # image = F.to_pil_image(image.squeeze(0))

    # Asynchronous copy to CPU
    # image = image.cpu()
    # torch.cuda.current_stream().synchronize()  # Ensure the copy is complete
    # image = image.permute(0, 2, 3, 1).float().numpy()
    # image = (image * 255).round().astype("uint8")[0]
    # image = Image.fromarray(image)

    print("Done with that")
    
    
    # Save the image
    image.save(filename)
    torch.cuda.empty_cache()