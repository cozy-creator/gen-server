from typing import Dict, Tuple, Union, List
from gen_server.types import CustomNode, NodeInterface, ImageOutputType
import random
import os
import blake3
from urllib.parse import urlparse
from PIL import Image
import numpy as np
from .paths import get_folder_path, get_save_image_path
import torch
import json
from typing import Optional
from PIL.PngImagePlugin import PngInfo
import boto3
from dotenv import load_dotenv
from .helper_decorators import convert_image_format
import io


load_dotenv()  # Load environment variables from .env file

env = os.getenv('ENVIRONMENT')
print(f"Environment: {env}")

# S3 configuration
s3_folder = os.getenv('S3__FOLDER', '')
s3_bucket_name = os.getenv('S3__BUCKET_NAME', '')
s3_endpoint_fqdn = os.getenv('S3__ENDPOINT_URL', '')
s3_access_key = os.getenv('S3__ACCESS_KEY', '')
s3_secret_key = os.getenv('S3__SECRET_ACCESS_KEY', '')

print(f"S3 Bucket: {s3_bucket_name}")

# Set up S3 client
if not all([s3_bucket_name, s3_endpoint_fqdn, s3_access_key, s3_secret_key]):
    raise ValueError('Missing S3 configuration in production mode')

s3 = boto3.client('s3',
                    endpoint_url=f'https://{s3_endpoint_fqdn}',
                    aws_access_key_id=s3_access_key,
                    aws_secret_access_key=s3_secret_key,
                    region_name=os.getenv("S3__REGION_NAME"))


class SaveFile(CustomNode):
    """
    Custom node to save or upload generated images.
    """

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

    def __call__(self, images: ImageOutputType, temp: bool = True, filename_prefix: str = "ComfyCreator", 
                 bucket_name: str = "my-bucket", prompt=None, extra_pnginfo=None) -> Dict[str, Union[str, None]]:
        """
        Saves or uploads the images based on the configuration.
        """
        results: List[Dict[str, Union[str, None]]] = []

        if env == "local":  # Assuming environment variables for local/production
            if temp:
                save_image_to_respective_path(self.temp_prefix_append,
                                              self.temp_dir, images, filename_prefix, prompt,
                                              extra_pnginfo, self.temp_compress_level, "temp", results)
            else:
                save_image_to_respective_path(self.prefix_append, self.output_dir, images,
                                              filename_prefix, prompt, extra_pnginfo,
                                              self.compress_level, self.type, results=[])
        elif env == "production":  # Assuming you have environment variables for local/prod
            if temp:
                # Upload the image to the "temp" folder
                temp_image_url = self.upload_to_s3(images, bucket_name, folder_name="temp")
                results.append({"output": {"temp_image_url": temp_image_url}})
            else:
                # Upload the image to the output folder
                image_url = self.upload_to_s3(images, bucket_name)
                results.append({"output": {"image_url": image_url}})

        return {"images": results}

    # @convert_image_format
    def upload_to_s3(self, image_data: Union[bytes, List[bytes]], bucket_name, folder_name: str = None):
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
            key = f'{folder_name}/{filename}' if folder_name else f'{os.getenv("S3__FOLDER")}/{filename}'

            # Upload the image data
            s3.put_object(Bucket=s3_bucket_name, Key=key, Body=img_bytes, ACL="public-read")

            # Generate and append the image URL
            endpoint_url = s3.meta.endpoint_url
            hostname = urlparse(endpoint_url).hostname
            image_url = f"https://{s3_bucket_name}.{hostname}/{key}"

            image_urls.append({
                "url": image_url,
                "filename": filename,
                "subfolder": folder_name if folder_name else "root",
                "type": folder_name if folder_name else "output"
            })

        print(image_urls)

        return image_urls if len(image_urls) > 1 else image_urls[0]

@convert_image_format
def save_image_to_respective_path(prefix_append, output_dir, 
                                      images: torch.Tensor, filename_prefix, prompt, 
                                      extra_pnginfo, compress_level,
                                      type, results):
        
        filename_prefix += prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = get_save_image_path(
            filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
        )
        for batch_number, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata: Optional[PngInfo] = None
            # if not args.disable_metadata:
            #     metadata = PngInfo()
            #     if prompt is not None:
            #         metadata.add_text("prompt", json.dumps(prompt))
            #     if extra_pnginfo is not None:
            #         for x in extra_pnginfo:
            #             metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=compress_level)
            results.append({"filename": file, "subfolder": subfolder, "type": type})
            counter += 1



