from multiprocessing import Pool
from typing import List
import os
from PIL.Image import Image
from gen_server.utils.paths import get_next_counter
import torch
from typing import Optional

from torchvision import transforms
from torchvision.transforms import ToPILImage
from gen_server.utils.paths import get_assets_dir

from typing import TypedDict


class FileUrl(TypedDict):
    url: str
    is_temp: bool


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


def tensor_to_pil(tensor: torch.Tensor) -> list[Image]:
    """
    Convert a batch of PyTorch tensors to a list of PIL Images.

    Parameters:
    - tensor: torch.Tensor - The tensor to convert. Assumes the tensor has a batch dimension.

    Returns:
    - list[PIL.Image.Image]: The list of tensors as PIL images.
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

    print(is_temp)

    assets_dir = get_assets_dir()
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
