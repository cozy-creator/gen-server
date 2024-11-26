import json
import time
from typing import Optional, Any
from cozy_runtime import Architecture, StateDict, TorchDevice, ComponentMetadata
from transformers import UMT5EncoderModel, UMT5Config
from diffusers.utils.import_utils import is_accelerate_available
from diffusers.models.model_loading_utils import load_model_dict_into_meta

import os
import re
import torch
import logging
from contextlib import nullcontext

from cozy_runtime.utils.device import get_torch_device

logger = logging.getLogger(__name__)

if is_accelerate_available():
    from accelerate import init_empty_weights

config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "aura_text_encoder_config.json"
)


def convert_auraflow_t5_checkpoint_to_diffusers(checkpoint: dict):
    keys = list(checkpoint.keys())
    text_model_dict = {}

    remove_prefixes = ["text_encoders.pile_t5xl.transformer."]

    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                diffusers_key = key.replace(prefix, "")
                text_model_dict[diffusers_key] = checkpoint.get(key)

    return text_model_dict


class AuraFlowTextEncoder(Architecture[UMT5EncoderModel]):
    """
    Architecture definition for the AuraFlow Text Encoder (T5-based).
    """

    def __init__(self, **ignored: Any):
        with open(config_path, "r") as file:
            # Create diffusers class
            config = json.load(file)
        text_encoder_config = UMT5Config.from_dict(config)
        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            text_encoder = UMT5EncoderModel(text_encoder_config)

        self._model = text_encoder
        self._config = config

        self._display_name = "AuraFlow Text Encoder"
        self._input_space = "AuraFlow"
        self._output_space = "AuraFlow"

    @classmethod
    def detect(  # type: ignore
        cls, state_dict: StateDict, **ignored: Any
    ) -> Optional[ComponentMetadata]:
        """
        Detects whether the given state dictionary matches the AuraFlow Text Encoder architecture.
        """
        state_key = "text_encoders.pile_t5xl.transformer.shared.weight"  # Check for a key specific to T5

        return (
            ComponentMetadata(
                display_name="AuraFlow Text Encoder",
                input_space="AuraFlow",
                output_space="AuraFlow",
            )
            if state_key in state_dict
            else None
        )

    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = "cpu"):
        """
        Loads the AuraFlow Text Encoder model from the given state dictionary.
        """
        print("Loading AuraFlow Text Encoder")
        start = time.time()

        text_encoder = self._model
        text_dict = {
            key: state_dict[key]
            for key in state_dict
            if key.startswith("text_encoders.pile_t5")
        }
        # print(f"text: {text_dict}")
        text_state_dict = convert_auraflow_t5_checkpoint_to_diffusers(text_dict)

        if is_accelerate_available():
            print("Using accelerate")
            unexpected_keys = load_model_dict_into_meta(
                text_encoder, text_state_dict, dtype=torch.float16
            )
            if text_encoder._keys_to_ignore_on_load_unexpected is not None:
                for pat in text_encoder._keys_to_ignore_on_load_unexpected:
                    unexpected_keys = [
                        k for k in unexpected_keys if re.search(pat, k) is None
                    ]

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint were not used when initializing {UMT5EncoderModel.__name__}: \n {[', '.join(unexpected_keys)]}"
                )

        else:
            text_encoder.load_state_dict(text_state_dict)
            text_encoder.to(torch.float16)

        # text_encoder.to(torch.float16)
        # text_encoder_3.to("cuda")
        # text_encoder.to_empty(device=get_torch_device())

        print(f"Loaded AuraFlow Text Encoder in {time.time() - start:.2f} seconds")
