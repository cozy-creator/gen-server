import json
import time
from typing import Dict, Optional
from gen_server import Architecture, StateDict, TorchDevice
from transformers import T5EncoderModel, T5Config
import safetensors
from diffusers.utils import is_accelerate_available
from diffusers.models.modeling_utils import load_model_dict_into_meta
from diffusers.loaders.single_file_utils import convert_sd3_t5_checkpoint_to_diffusers
import os
import re
import torch
import logging
from contextlib import nullcontext

logger = logging.getLogger(__name__)


if is_accelerate_available():
    from accelerate import init_empty_weights


config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

class SD3TextEncoder3(Architecture[T5EncoderModel]):
    """
    Architecture definition for the SD3 Text Encoder 3 (T5-based).
    """

    def __init__(self):
        with open(config_path, 'r') as file:
            # Create diffusers class
            config = json.load(file)
        text_encoder_config = T5Config.from_dict(config)
        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            text_encoder = T5EncoderModel(text_encoder_config)
        super().__init__(
            model=text_encoder,  # Initialize with config
            config=text_encoder_config,
            input_space="SD3",
            output_space="SD3"
        )

    @classmethod
    def detect(cls, state_dict: StateDict) -> bool:
        """
        Detects whether the given state dictionary matches the SD3 Text Encoder 3 architecture.
        """
        return "text_encoders.t5xxl.transformer.shared.weight" in state_dict  # Check for a key specific to T5

    def load(self, state_dict: StateDict, device: TorchDevice = None):
        """
        Loads the SD3 Text Encoder 3 model from the given state dictionary.
        """
        print("Loading SD3 Text Encoder 3")
        start = time.time()

        text_encoder_3 = self.model
        text_dict = {key: state_dict[key] for key in state_dict if key.startswith("text_encoders.t5xxl.")}
        # print(f"text: {text_dict}")
        text_state_dict = convert_sd3_t5_checkpoint_to_diffusers(text_dict)

        if is_accelerate_available():
            unexpected_keys = load_model_dict_into_meta(text_encoder_3, text_state_dict, dtype=torch.float16)
            if text_encoder_3._keys_to_ignore_on_load_unexpected is not None:
                for pat in text_encoder_3._keys_to_ignore_on_load_unexpected:
                    unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint were not used when initializing {T5EncoderModel.__name__}: \n {[', '.join(unexpected_keys)]}"
                )

        else:
            text_encoder_3.load_state_dict(text_state_dict)
            text_encoder_3.to(torch.float16)
            
        text_encoder_3.to("cuda")
