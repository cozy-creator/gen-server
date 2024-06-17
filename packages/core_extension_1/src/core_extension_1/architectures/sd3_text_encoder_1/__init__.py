import json
import time
from typing import Dict, Optional
from gen_server import Architecture, StateDict, TorchDevice
from transformers import CLIPTextModelWithProjection, CLIPTextConfig
import safetensors
from diffusers.utils import is_accelerate_available
from diffusers.models.modeling_utils import load_model_dict_into_meta
import re
import os
import torch
import logging
from contextlib import nullcontext

logger = logging.getLogger(__name__)

if is_accelerate_available():
    from accelerate import init_empty_weights

LDM_CLIP_PREFIX_TO_REMOVE = [
    "cond_stage_model.transformer.",
    "conditioner.embedders.0.transformer.",
    "text_encoders.clip_l.transformer.",
]

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

class SD3TextEncoder1(Architecture[CLIPTextModelWithProjection]):
    """
    Architecture definition for the SD3 Text Encoder 1 (CLIP-based).
    """

    def __init__(self):
        with open(config_path, 'r') as file:
            # Create diffusers class
            config = json.load(file)
        text_encoder_config = CLIPTextConfig.from_dict(config)
        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            text_encoder = CLIPTextModelWithProjection(text_encoder_config)
        super().__init__(
            model=text_encoder,
            config=text_encoder_config,
            input_space="SD3",
            output_space="SD3"
        )

    @classmethod
    def detect(cls, state_dict: StateDict) -> bool:
        """
        Detects whether the given state dictionary matches the SD3 Text Encoder 1 architecture.
        """
        return "text_encoders.clip_l.transformer.text_model.embeddings.position_embedding.weight" in state_dict

    def load(self, state_dict: StateDict, device: TorchDevice = None):
        """
        Loads the SD3 Text Encoder 1 model from the given state dictionary.
        """
        print("Loading SD3 Text Encoder 1")
        start = time.time()

        text_encoder = self.model
        text_encoder_state_dict = {key: state_dict[key] for key in state_dict if key.startswith("text_encoders.clip_l.")}

        remove_prefixes = LDM_CLIP_PREFIX_TO_REMOVE
        keys = list(text_encoder_state_dict.keys())
        text_model_dict = {}

        for key in keys:
            for prefix in remove_prefixes:
                if key.startswith(prefix):
                    diffusers_key = key.replace(prefix, "")
                    text_model_dict[diffusers_key] = text_encoder_state_dict[key]

        if is_accelerate_available():
            torch_dtype = next(text_encoder.parameters()).dtype
            unexpected_keys = load_model_dict_into_meta(text_encoder, text_model_dict, dtype=torch_dtype)
            if text_encoder._keys_to_ignore_on_load_unexpected is not None:
                for pat in text_encoder._keys_to_ignore_on_load_unexpected:
                    unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint were not used when initializing {text_encoder.__class__.__name__}: \n {[', '.join(unexpected_keys)]}"
                )

        else:
            if not (hasattr(text_encoder, "embeddings") and hasattr(text_encoder.embeddings.position_ids)):
                text_model_dict.pop("text_model.embeddings.position_ids", None)

            text_encoder.load_state_dict(text_model_dict, strict=False)
            text_encoder.to(torch.float16)

        text_encoder.to_empty(device=torch.device("cuda"))