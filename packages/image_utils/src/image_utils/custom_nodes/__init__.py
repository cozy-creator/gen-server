from typing import Dict, Tuple, Union, List
from gen_server.base_types import CustomNode, NodeInterface, ImageOutputType
import random
import os
import blake3
from urllib.parse import urlparse
from PIL.Image import Image
import numpy as np
from .paths import get_folder_path, get_next_counter
import torch
import json
from typing import Optional
from PIL.PngImagePlugin import PngInfo
import boto3
# from dotenv import load_dotenv
from .helper_decorators import convert_image_format
import io
from torchvision import transforms
from torchvision.transforms import ToPILImage
from gen_server.globals import comfy_config
from multiprocessing import Pool
from typing import TypedDict


# load_dotenv()  # Load environment variables from .env file

# env = os.getenv('ENVIRONMENT')
# print(f"Environment: {env}")

# S3 configuration
# s3_folder = os.getenv('S3__FOLDER', '')
# s3_bucket_name = os.getenv('S3__BUCKET_NAME', '')
# s3_endpoint_fqdn = os.getenv('S3__ENDPOINT_URL', '')
# s3_access_key = os.getenv('S3__ACCESS_KEY', '')
# s3_secret_key = os.getenv('S3__SECRET_ACCESS_KEY', '')

# print(f"S3 Bucket: {s3_bucket_name}")

# Set up S3 client
# if not all([s3_bucket_name, s3_endpoint_fqdn, s3_access_key, s3_secret_key]):
#     raise ValueError('Missing S3 configuration in production mode')

# s3 = boto3.client('s3',
#                     endpoint_url=f'https://{s3_endpoint_fqdn}',
#                     aws_access_key_id=s3_access_key,
#                     aws_secret_access_key=s3_secret_key,
#                     region_name=os.getenv("S3__REGION_NAME"))

class FileUrl(TypedDict):
    url: str
    is_temp: bool

class SaveFile(CustomNode):
    """
    Custom node to save or upload generated images.
    """
    display_name = "Saves Images"

    category = "images"

    description = "Useful for saving images"

    def __init__(self) -> None:
        self.output_dir = get_folder_path("output")
        self.temp_dir = get_folder_path("temp")
        self.type = "output"
        self.prefix_append = ""
        self.temp_prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
        self.compress_level = 4
        self.temp_compress_level = 1

    @staticmethod
    def update_interface(inputs: Dict[str, Union[Tuple[str, ...], Tuple[str, Dict[str, Union[str, int]]]]] = None) -> NodeInterface:
        """
        Defines the input and output interface for the node.
        """
        interface = {
            "inputs": {
                "images": bytes,
                "temp": bool,
                # "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                # "save_type": (["local", "s3"], {"default": "local"}),
                # "bucket_name": ("STRING", {"default": "my-bucket"}),
            },
            "outputs": {},  # No explicit outputs for this node
        }

        if inputs:
            # Check if the node is in "temp" mode
            if inputs.get("temp", False):
                interface["outputs"]["temp_image_url"] = str

            # Check if the node is in "save" mode and environment is prod
            elif inputs.get("temp", False) is False and "PROD" in os.environ:
                interface["outputs"]["image_url"] = str

        return interface

    def __call__(
            self,
            # TO DO: make this type less fixed; ImageOutputType is also needed
            images: List[Image],
            # images: torch.Tensor,
            temp: bool = True,
            filename_prefix: str = "ComfyCreator", 
            bucket_name: str = "my-bucket",
            image_metadata: Optional[dict] = None
        ) -> Dict[str, List[FileUrl]]:
        """
        Saves images to the local filesystem or a remote S3 bucket.
        """

        if comfy_config.filesystem_type == "LOCAL":  # Assuming environment variables for local/production
            if temp:
                prefix = self.temp_prefix_append
                dir = self.temp_dir
            else:
                prefix = self.prefix_append
                dir = self.output_dir
                
            image_urls = save_image_to_filesystem(
                images,
                compress_level=self.compress_level,
                image_metadata=image_metadata,
                is_temp=temp)
            
            # prepend our server's GET-file endpoint to these relative paths
            for url in image_urls:
                print(url["url"])
                url["url"] = f"http://{comfy_config.host}:{comfy_config.port}/files/{url['url']}"
            
        elif comfy_config.filesystem_type == "S3":
            # Assuming you have environment variables for local/prod
            # Upload the image to the "temp" folder
            image_urls = self.upload_to_s3(
                images, bucket_name, folder_name="temp" if temp else None)
        
        else:
            image_urls = []

        return { "images": image_urls }

    # @convert_image_format
    def upload_to_s3(self, image_data: Union[bytes, List[bytes]], bucket_name, folder_name: Optional[str] = None):
        """
        Uploads image data to an S3 bucket and returns the URL(s) of the uploaded image(s).

        Args:
            image_data (Union[bytes, List[bytes]]): A byte string or a list of byte strings representing image(s) to be uploaded.
            bucket_name (str): Name of the S3 bucket.
            folder_name (str): Optional folder name within the bucket to upload the image(s) to.

        Returns:
            Union[str, List[str]]: A single URL or a list of URLs of the uploaded image(s).
        """
        if not isinstance(image_data, list):
            image_data = [image_data]

        image_urls = []

        for idx, img_pil in enumerate(image_data):
            with io.BytesIO() as output:
                img_pil.save(output, format="PNG")
                img_bytes = output.getvalue()

            filename = f"{blake3.blake3(img_bytes).hexdigest()}.png"
            
            key = f'{comfy_config.s3["folder"]}/{filename}' if comfy_config.s3["folder"] else f'{filename}'
            

            # Upload the image data
            comfy_config.s3["client"].put_object(Bucket=comfy_config.s3["bucket_name"], Key=key, Body=img_bytes, ACL="public-read")

            # Generate and append the image URL
            # endpoint_url = s3.meta.endpoint_url
            # hostname = urlparse(endpoint_url).hostname
            # image_url = f"https://{s3_bucket_name}.{hostname}/{key}"
            image_url = f"{comfy_config.s3['url']}/{key}"

            image_urls.append({
                "url": image_url,
                "filename": filename,
                "subfolder": folder_name if folder_name else "root",
                "type": folder_name if folder_name else "output"
            })

        print(image_urls)

        return image_urls if len(image_urls) > 1 else image_urls[0]


def pil_to_tensor(image: Image) -> torch.Tensor:
    """
    Convert a PIL Image to a PyTorch tensor.

    Parameters:
    - image: PIL.Image.Image - The image to convert.

    Returns:
    - torch.Tensor: The image as a tensor.
    """
    transform = transforms.ToTensor()
    tensor = transform(image)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> List[Image]:
    """
    Convert a batch of PyTorch tensors to a list of PIL Images.

    Parameters:
    - tensor: torch.Tensor - The tensor to convert. Assumes the tensor has a batch dimension.

    Returns:
    - List[PIL.Image.Image]: The list of tensors as PIL images.
    """
    transform = ToPILImage()
    images = [transform(t) for t in tensor]
    return images


def save_image_to_filesystem(
        images: List[Image],
        # tensor_images: torch.Tensor,
        filename_prefix: str = '',
        image_metadata: Optional[dict] = None,
        compress_level: Optional[int] = None,
        file_type: str = "png",
        is_temp: bool = False
    ) -> List[FileUrl]:
    # TO DO: make this conversion more automatic in the executor rather than manual / fixed
    # images = tensor_to_pil(tensor_images)
    
    # TO DO: move this logic somewhere more general
    workspace_dir = comfy_config.workspace_dir
    if not workspace_dir:
        raise FileNotFoundError(f"The workspace directory '{workspace_dir}' does not exist.")

    print(is_temp)
    
    assets_dir = os.path.join(
        workspace_dir, 'assets', 'temp' if is_temp else '')
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    # Get the next counter
    counter = get_next_counter(assets_dir, filename_prefix)

    # Prepare data for multiprocessing
    image_data_list = [
        (batch_number, image, filename_prefix, file_type, assets_dir, image_metadata, compress_level, is_temp, counter + batch_number)
        for batch_number, image in enumerate(images)
    ]

    # Use multiprocessing to save images in parallel
    with Pool() as pool:
        urls = pool.map(save_image, image_data_list)

    return urls


def save_image(image_data) -> FileUrl:
    batch_number, image, filename_prefix, file_type, assets_dir, image_metadata, compress_level, is_temp, counter = image_data
    filename_with_batch_num = filename_prefix.replace("%batch_num%", str(batch_number))
    file_name = f"{filename_with_batch_num}_{counter:05}_.{file_type}"
    image_path = os.path.join(assets_dir, file_name)
    
    image.save(image_path, pnginfo=image_metadata, compress_level=compress_level)
    
    # TO DO: we need to sychronize this with how they're read from the get-files endpoint
    return FileUrl(url=file_name, is_temp=is_temp)

# def save_image_to_filesystem(
#         tensor_images: torch.Tensor,
#         filename_prefix: str = "",
#         image_metadata: Optional[dict] = None,
#         compress_level: Optional[int] = None,
#         file_type: Optional[str] = "png",
#         is_temp: bool = False
#     ) -> List[FileUrl]:
#     # TO DO: make this conversion more flexible
#     images = tensor_to_pil(tensor_images)
    
#     workspace_dir = comfy_config.workspace_dir
#     if not workspace_dir:
#         raise FileNotFoundError(f"The workspace directory '{workspace_dir}' does not exist.")
    
#     assets_dir = os.path.join(workspace_dir, 'assets')
    
#     if is_temp:
#         assets_dir = os.path.join(assets_dir, 'temp')
    
#     if not os.path.exists(assets_dir):
#         os.makedirs(assets_dir)
    
#     counter = 0
#     urls: List[FileUrl] = []
    
#     for batch_number, image in enumerate(images):
#         filename_with_batch_num = filename_prefix.replace("%batch_num%", str(batch_number))
#         file = f"{filename_with_batch_num}_{counter:05}_.{file_type}"
#         image_path = os.path.join(assets_dir, file)
#         image.save(image_path, pnginfo=image_metadata, compress_level=compress_level)
#         urls.append(FileUrl(url=image_path, is_temp=is_temp))
#         counter += 1
    
#     return urls
    
#     full_output_folder, filename, counter, subfolder, filename_prefix = get_save_image_path(
#         filename_prefix, output_dir, images[0].width, images[0].height
#     )
    
#     urls: List[FileUrl] = []
    
#     for batch_number, image in enumerate(images):
#         # i = 255. * image.cpu().numpy()
#         # img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
#         metadata: Optional[PngInfo] = None
#         # if not args.disable_metadata:
#         #     metadata = PngInfo()
#         #     if prompt is not None:
#         #         metadata.add_text("prompt", json.dumps(prompt))
#         #     if extra_pnginfo is not None:
#         #         for x in extra_pnginfo:
#         #             metadata.add_text(x, json.dumps(extra_pnginfo[x]))
        
#         filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
#         file = f"{filename_with_batch_num}_{counter:05}_.png"
#         image.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=compress_level)
#         urls.append(FileUrl(url=file, is_temp=is_temp))
#         counter += 1
    
#     return urls

