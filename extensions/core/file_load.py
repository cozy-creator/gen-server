# import numpy as np
# import torch
# import io
# from PIL import Image, ImageOps, ImageSequence
# import blake3
# import os
# import paths
# from paths import s3_folder, s3_bucket_name, s3
# from paths import env



# class LoadImage:

#     def __init__(self):
#         self.temp_dir = paths.get_folder_path('temp')

#     @classmethod
#     def INPUT_TYPES(s):
#         input_dir = paths.get_folder_path('input')
#         files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
#         return {"required":
#                     {"image": (sorted(files), {"image_upload": True})},
#                 }

#     CATEGORY = "image"

#     RETURN_TYPES = ("IMAGE", "MASK")
#     FUNCTION = "load_image"

#     def load_image(self, image):
        
#         # Check environment to know where to fetch files from
#         print(f'Environment: {env}')
#         if env == 'LOCAL':  
#             image_path = paths.get_annotated_filepath(image)
#             img = Image.open(image_path)
#         elif env == 'PROD':
#             file_key = f'{s3_folder}/{image}' if s3_folder else image
#             cached_file = os.path.join(self.temp_dir, image)
#             print(cached_file)

#             if os.path.exists(cached_file):
#                 img = Image.open(cached_file)
#             else:
#                 try:
#                     print(f'Fetching image from S3 bucket: {s3_bucket_name}')
#                     s3_object = s3.get_object(Bucket=s3_bucket_name, Key=file_key)
                    
#                     file_data = s3_object['Body'].read()
#                     img = Image.open(io.BytesIO(file_data))
#                     img.save(cached_file)
#                 except s3.exceptions.NoSuchKey:
#                     raise FileNotFoundError(f'File not found in S3 bucket: {file_key}')
#         else:
#             raise ValueError("Please specify a valid environment name")


        
#         output_images = []
#         output_masks = []
#         for i in ImageSequence.Iterator(img):
#             i = ImageOps.exif_transpose(i)
#             if i.mode == 'I':
#                 i = i.point(lambda i: i * (1 / 255))
#             image = i.convert("RGB")
#             image = np.array(image).astype(np.float32) / 255.0
#             image = torch.from_numpy(image)[None,]
#             if 'A' in i.getbands():
#                 mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
#                 mask = 1. - torch.from_numpy(mask)
#             else:
#                 mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
#             output_images.append(image)
#             output_masks.append(mask.unsqueeze(0))

#         if len(output_images) > 1:
#             output_image = torch.cat(output_images, dim=0)
#             output_mask = torch.cat(output_masks, dim=0)
#         else:
#             output_image = output_images[0]
#             output_mask = output_masks[0]

#         return (output_image, output_mask)

#     @classmethod
#     def IS_CHANGED(s, image):
#         image_path = paths.get_annotated_filepath(image)
#         with open(image_path, 'rb') as f:
#             file_content = f.read()

#             # Open the image using PIL
#             image = Image.open(io.BytesIO(file_content))

#             # Get the pixel data
#             pixel_data = image.tobytes()

#             hash = blake3.blake3(pixel_data).hexdigest()
            
#         return hash

#     @classmethod
#     def VALIDATE_INPUTS(s, image):
#         if env == 'LOCAL':
#             if not paths.exists_annotated_filepath(image):
#                 return "Invalid image file: {}".format(image)

#         return True