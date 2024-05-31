# from typing import Dict, List, Tuple, Union, Optional
# import random
# import torch
# from PIL import Image
# import os
# import json
# import paths
# from PIL.PngImagePlugin import PngInfo
# import numpy as np
# from dotenv import load_dotenv
# from extensions.core.helper.helper_decorators import convert_image_format
# import blake3
# import boto3
# from urllib.parse import urlparse
# from botocore.exceptions import ClientError
# from core_library.uploader import S3Uploader
# from gen_server.settings import settings
# from paths import get_save_image_path


# load_dotenv()

# env = os.getenv("ENV")


# # S3 configuration
# s3_folder = os.getenv('S3_FOLDER', '')
# s3_bucket_name = os.getenv('S3_BUCKET_NAME', '')
# s3_endpoint_fqdn = os.getenv('S3_ENDPOINT_FQDN', '')
# s3_access_key = os.getenv('S3_ACCESS_KEY', '')
# s3_secret_key = os.getenv('S3_SECRET_KEY', '')

# print(f"S3 Bucket: {s3_bucket_name}")

# # Set up S3 client
# if not all([s3_bucket_name, s3_endpoint_fqdn, s3_access_key, s3_secret_key]):
#     raise ValueError('Missing S3 configuration in production mode')

# s3 = boto3.client('s3',
#                     endpoint_url=f'https://{s3_endpoint_fqdn}',
#                     aws_access_key_id=s3_access_key,
#                     aws_secret_access_key=s3_secret_key,
#                     region_name=os.getenv("REGION_NAME"))
# try:
#     policy_status = s3.put_bucket_lifecycle_configuration(
#             Bucket=s3_bucket_name,
#             LifecycleConfiguration={
#                     'Rules': 
#                         [
#                             {
#                             'Expiration':
#                                 {
#                                 'Days': 1,
#                                 'ExpiredObjectDeleteMarker': True
#                                 },
#                             'Prefix': 'temp/',
#                             'Filter': {
#                             'Prefix': 'temp/',
#                             },
#                             'Status': 'Enabled'
#                             }
#                         ]})
# except ClientError as e:
#     print("Unable to apply bucket policy. \nReason:{0}".format(e))



# def save_image_to_respective_path(prefix_append, output_dir, 
#                                       images: torch.Tensor, filename_prefix, prompt, 
#                                       extra_pnginfo, compress_level,
#                                       type, results):
        
#         filename_prefix += prefix_append
#         full_output_folder, filename, counter, subfolder, filename_prefix = paths.get_save_image_path(
#             filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
#         )
#         for batch_number, image in enumerate(images):
#             i = 255. * image.cpu().numpy()
#             img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
#             metadata: Optional[PngInfo] = None
#             # if not args.disable_metadata:
#             #     metadata = PngInfo()
#             #     if prompt is not None:
#             #         metadata.add_text("prompt", json.dumps(prompt))
#             #     if extra_pnginfo is not None:
#             #         for x in extra_pnginfo:
#             #             metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            
#             filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
#             file = f"{filename_with_batch_num}_{counter:05}_.png"
#             img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=compress_level)
#             results.append({"filename": file, "subfolder": subfolder, "type": type})
#             counter += 1


# class SaveFile:
#     def __init__(self) -> None:
#         self.output_dir = paths.get_folder_path("output")
#         self.temp_dir = paths.get_folder_path("temp")
#         self.type = "output"
#         self.prefix_append = ""
#         self.temp_prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
#         self.compress_level = 4
#         self.temp_compress_level = 1

#     @classmethod
#     def INPUT_TYPES(s) -> Dict[str, Union[Tuple[str, ...], Tuple[str, Dict[str, Union[str, int]]]]]:
#         return {
#             "required": {
#                 "images": ("IMAGE", ),
#                 # "filename_prefix": ("STRING", {"default": "ComfyUI"}),
#                 "temp": ("BOOLEAN", {"default": True}),
#                 # "save_type": (["local", "s3"], {"default": "local"}),
#                 # "bucket_name": ("STRING", {"default": "my-bucket"}),
#             },
#             "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
#         }

#     RETURN_TYPES: Tuple = ()
#     FUNCTION = "save_and_preview_images"

#     OUTPUT_NODE = True

#     CATEGORY = "image"


#     def save_and_preview_images(
#         self,
#         images: List[Union[torch.Tensor, bytes]],
#         temp: bool = True,
#         filename_prefix: str = "ComfyUI",
#         # save_images: bool = False,
#         # save_type: str = "local",
#         bucket_name: str = "my-bucket",
#         prompt = None,
#         extra_pnginfo = None,
#     ):
        
#         results: List[Dict[str, Union[str, None]]] = []


#         if env == "LOCAL":
#             if temp:
#                 save_image_to_respective_path(self.temp_prefix_append, 
#                                       self.temp_dir, images, filename_prefix, prompt, 
#                                       extra_pnginfo, self.temp_compress_level, "temp", results)
#             else:
#                 save_image_to_respective_path(self.prefix_append, self.output_dir, images, 
#                                           filename_prefix, prompt, extra_pnginfo, 
#                                           self.compress_level, self.type, results=[])
#         elif env == "PROD":
#             if temp:
#                 # Upload the image to the "temp" folder
#                 temp_image_url = self.upload_to_s3(images, bucket_name, folder_name="temp")

#                 results[-1]["output"] = {"temp_image_url": temp_image_url}

#             elif isinstance(images, bytes) or isinstance(images, torch.Tensor):
#                 print("Here 2")
#                 image_url = self.upload_to_s3(images, bucket_name)
#                 # print(image_url )

#                 if results:
#                     results[-1]["output"] = {"image_url": image_url}
#                 else:
#                     print("Here 3")
#                     # results.append({"output": {"image_url": image_url}})
#                     results = image_url

#         yield {"ui": {"images": results}}

    
#     @convert_image_format
#     def upload_to_s3(self, image_data: Union[bytes, List[bytes]], bucket_name, folder_name: str = None):
#         """
#         Uploads image data to an S3 bucket and returns the URL(s) of the uploaded image(s).

#         Args:
#             image_data (Union[bytes, List[bytes]]): A byte string or a list of byte strings representing image(s) to be uploaded.
#             bucket_name (str): Name of the S3 bucket.
#             folder_name (str): Optional folder name within the bucket to upload the image(s) to.

#         Returns:
#             Union[str, List[str]]: A single URL or a list of URLs of the uploaded image(s).
#         """

#         # Initialize S3Uploader with configuration
#         uploader = S3Uploader(settings.s3)


#         if not isinstance(image_data, list):
#             image_data = [image_data]

#         image_urls = []

#         for idx, img_bytes in enumerate(image_data):
#             filename = f"{blake3.blake3(img_bytes).hexdigest()}.png"
#             key = f'{folder_name}/{filename}' if folder_name else f'{os.getenv("S3_FOLDER")}/{filename}'

#              # Upload the image using S3Uploader
#             uploader.upload(img_bytes, key, "image/png")

#             # Get the uploaded image URL
#             image_url = uploader.get_uploaded_file_url(key)

#             image_urls.append({
#                 "url": image_url,
#                 "filename": filename,
#                 "subfolder": folder_name if folder_name else "root",
#                 "type": folder_name if folder_name else "output"
#             })

#         print(image_urls)

#         return image_urls if len(image_urls) > 1 else image_urls[0]