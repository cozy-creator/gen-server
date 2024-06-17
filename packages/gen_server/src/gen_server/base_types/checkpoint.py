from typing import Dict, Type
from unittest.mock import Base
from .architecture import Architecture


# key in dictionary needs to be unique; we need a unique identifier
# maybe internal identifier? Randomly generated? maybe full file path?
# maybe also have file-sycnhronization
# this will be different than what we send to the client
# "sdxl_base.fp16.safetensors" ->
# {
#     components: {
#         'vae': Vae_architecture,
#         'text_encoder': CLIP,
#         'text_decoder_2': CLIPWithPojection
#         'unet': SDXLUnet
#     },
#      file_path: '~/.comfy-creator/models/sdxl_base.fp16.safetensors',
#     dtype: torch.dtype.fp16,
#     group: 'sdxl',
#     display_name: 'SDXL Base',
#     author: ,
#     date: ,
#     (other metadata taken from the header)
# }

# SDXL models:
#     - SDXL Base
#     - Dreamshaper xlogy
# SD3 models:
#     - Base
#     - Fine Tuned 5


class Checkpoint:
    """
    This comfy-creator-specific metadata for a pretrained model / checkpoint file
    Does PyTorch already have something along these lines?
    """

    def __init__(self):
        self._components: Dict[str, Architecture] = {}
        self._display_name: str = 'something'

    @property
    def components(self) -> Dict[str, Architecture]:
        return self._components
