from typing import Dict, Tuple, Union, List
from gen_server import Category, Language
from gen_server.base_types import CustomNode, NodeInterface
import random
import os
import blake3
from PIL.Image import Image
from gen_server.utils.file_handler import get_file_handler
from .paths import get_folder_path, get_next_counter
import torch
from typing import Optional

import io
from torchvision import transforms
from torchvision.transforms import ToPILImage
from gen_server.config import get_config
from multiprocessing import Pool
from typing import TypedDict


class FileUrl(TypedDict):
    url: str
    is_temp: bool


class SaveFile(CustomNode):
    """
    Custom node to save or upload generated images.
    """

    display_name = {Language.ENGLISH: "Saves Images"}

    category = Category.INPAINTING

    description = {Language.ENGLISH: "Useful for saving images"}

    def __init__(self) -> None:
        self.output_dir = get_folder_path("output")
        self.temp_dir = get_folder_path("temp")
        self.type = "output"
        self.prefix_append = ""
        self.temp_prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5)
        )
        self.compress_level = 4
        self.temp_compress_level = 1

    @staticmethod
    def update_interface(
        inputs: Optional[
            Dict[str, Union[Tuple[str, ...], Tuple[str, Dict[str, Union[str, int]]]]]
        ] = None,
    ) -> NodeInterface:
        """
        Defines the input and output interface for the node.
        """
        interface = {
            "inputs": {
                "images": bytes,
                "temp": bool,
                "filename_prefix": ("STRING", {"default": "Cozy"}),
                "save_type": (["local", "s3"], {"default": "local"}),
                "bucket_name": ("STRING", {"default": "my_bucket"}),
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

    class NodeResponse(TypedDict):
        image_urls: List[FileUrl]

    def __call__(self, images: List[Image]) -> NodeResponse:
        """
        Saves images to the local filesystem or a remote S3 bucket.
        """

        image_urls = self.upload(images)
        return {"image_urls": image_urls}

    # @convert_image_format
    def upload(self, images: Union[List[Image], Image]):
        """
        Uploads image(s) data returns the URL(s) of the uploaded image(s).

        Args:
            image_data (Union[bytes, List[bytes]]): A byte string or a list of byte strings representing image(s) to be uploaded.

        Returns:
            Union[str, List[str]]: A single URL or a list of URLs of the uploaded image(s).
        """
        if not isinstance(images, list):
            images = [images]

        image_urls = []

        for img_pil in images:
            with io.BytesIO() as output:
                img_pil.save(output, format="PNG")
                img_bytes = output.getvalue()

            filename = f"{blake3.blake3(img_bytes).hexdigest()}.png"

            file_handler = get_file_handler(get_config())
            url = file_handler.upload_file(img_bytes, filename)
            image_urls.append({"url": url, "filename": filename})

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
    filename_prefix: str = "",
    image_metadata: Optional[dict] = None,
    compress_level: Optional[int] = None,
    file_type: str = "png",
    is_temp: bool = False,
) -> List[FileUrl]:
    # TO DO: make this conversion more automatic in the executor rather than manual / fixed
    # images = tensor_to_pil(tensor_images)

    # TO DO: move this logic somewhere more general
    workspace_path = get_config().workspace_path
    if not workspace_path:
        raise FileNotFoundError(
            f"The workspace directory '{workspace_path}' does not exist."
        )

    print(is_temp)

    assets_dir = os.path.join(workspace_path, "assets", "temp" if is_temp else "")
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    # Get the next counter
    counter = get_next_counter(assets_dir, filename_prefix)

    # Prepare data for multiprocessing
    image_data_list = [
        (
            batch_number,
            image,
            filename_prefix,
            file_type,
            assets_dir,
            image_metadata,
            compress_level,
            is_temp,
            counter + batch_number,
        )
        for batch_number, image in enumerate(images)
    ]

    # Use multiprocessing to save images in parallel
    with Pool() as pool:
        urls = pool.map(save_image, image_data_list)

    return urls


def save_image(image_data) -> FileUrl:
    (
        batch_number,
        image,
        filename_prefix,
        file_type,
        assets_dir,
        image_metadata,
        compress_level,
        is_temp,
        counter,
    ) = image_data
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

#     workspace_path = get_config().workspace_path
#     if not workspace_path:
#         raise FileNotFoundError(f"The workspace directory '{workspace_path}' does not exist.")

#     assets_dir = os.path.join(workspace_path, 'assets')

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
