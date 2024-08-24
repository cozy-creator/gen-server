import json
import time
from typing import Optional, Any
from gen_server import Architecture, StateDict, TorchDevice, ComponentMetadata
from transformers import T5EncoderModel, T5Config
from diffusers.utils.import_utils import is_accelerate_available
from diffusers.models.model_loading_utils import load_model_dict_into_meta

# from diffusers.loaders.single_file_utils import convert_sd3_t5_checkpoint_to_diffusers
import os
import re
import torch
import logging
from contextlib import nullcontext

from gen_server.utils.device import get_torch_device

logger = logging.getLogger(__name__)

if is_accelerate_available():
    from accelerate import init_empty_weights

config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config_text_encoder_3.json"
)


def convert_sd3_t5_checkpoint_to_diffusers(checkpoint: StateDict):
    keys = list(checkpoint.keys())
    text_model_dict = {}

    remove_prefixes = ["text_encoders.t5xxl.transformer."]

    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                diffusers_key = key.replace(prefix, "")
                text_model_dict[diffusers_key] = checkpoint.get(key)

    return text_model_dict


class SD3TextEncoder3(Architecture[T5EncoderModel]):
    """
    Architecture definition for the SD3 Text Encoder 3 (T5-based).
    """

    def __init__(self, **ignored: Any):
        with open(config_path, "r") as file:
            # Create diffusers class
            config = json.load(file)
        text_encoder_config = T5Config.from_dict(config)
        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            text_encoder = T5EncoderModel(text_encoder_config)

        self._model = text_encoder
        self._config = config

        self._display_name = "T5 XXL Text Encoder"
        self._input_space = "SD3"
        self._output_space = "SD3"

    @classmethod
    def detect( # type: ignore
        cls, state_dict: StateDict, **ignored: Any
    ) -> Optional[ComponentMetadata]:
        """
        Detects whether the given state dictionary matches the SD3 Text Encoder 3 architecture.
        """
        state_key = "text_encoders.t5xxl.transformer.shared.weight"  # Check for a key specific to T5

        return (
            ComponentMetadata(
                display_name="T5 XXL Text Encoder",
                input_space="SD3",
                output_space="SD3",
            )
            if state_key in state_dict
            else None
        )

    def load(self, state_dict: StateDict, device: TorchDevice = "cpu"): # type: ignore
        """
        Loads the SD3 Text Encoder 3 model from the given state dictionary.
        """
        print("Loading SD3 Text Encoder 3")
        start = time.time()

        text_encoder_3 = self._model
        text_dict = {
            key: state_dict[key]
            for key in state_dict
            if key.startswith("text_encoders.t5xxl.")
        }
        # print(f"text: {text_dict}")
        text_state_dict = convert_sd3_t5_checkpoint_to_diffusers(text_dict)

        # print(text_state_dict.keys())

        if is_accelerate_available():
            print("Using accelerate")
            torch_dtype = next(text_encoder_3.parameters()).dtype
            unexpected_keys = load_model_dict_into_meta(
                text_encoder_3, text_state_dict, dtype=torch.bfloat16
            )
            if text_encoder_3._keys_to_ignore_on_load_unexpected is not None:
                for pat in text_encoder_3._keys_to_ignore_on_load_unexpected:
                    unexpected_keys = [
                        k for k in unexpected_keys if re.search(pat, k) is None
                    ]

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint were not used when initializing {T5EncoderModel.__name__}: \n {[', '.join(unexpected_keys)]}"
                )

        else:
            text_encoder_3.load_state_dict(text_state_dict)
            text_encoder_3.to(torch.float16)

        # text_encoder_3.to_empty(device=get_torch_device())

        # text_encoder_3.to("cuda")
