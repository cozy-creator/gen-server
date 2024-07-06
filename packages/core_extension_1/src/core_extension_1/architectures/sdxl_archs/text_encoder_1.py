import os
import json
import time
from typing import Any, Optional

from typing_extensions import override
from gen_server import Architecture, StateDict, TorchDevice, ComponentMetadata
from transformers import CLIPTextModel, CLIPTextConfig
import torch
from diffusers.utils.import_utils import is_accelerate_available
from contextlib import nullcontext
from diffusers.models.model_loading_utils import load_model_dict_into_meta
import re
import logging

logger = logging.getLogger(__name__)

if is_accelerate_available():
    from accelerate import init_empty_weights

LDM_CLIP_PREFIX_TO_REMOVE = [
    "cond_stage_model.transformer.",
    "conditioner.embedders.0.transformer.",
    "text_encoders.clip_l.transformer.",
]

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_text_encoder_1.json")


class SDXLTextEncoder(Architecture[CLIPTextModel]):
    """
    The CLIP text-encoder used for the SDXL pipeline
    """

    def __init__(self, **ignored: Any):
        with open(config_path, "r") as file:
            config = json.load(file)

        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            text_encoder_config = CLIPTextConfig.from_dict(config)
            text_encoder = CLIPTextModel(text_encoder_config)

            self._model = text_encoder
            self._config = config
        
        self._display_name = "CLIP Text Encoder"
        self._input_space = "SDXL"
        self._output_space = "SDXL"

    @classmethod
    def detect(
        cls,
        state_dict: StateDict,
        **ignored: Any
    ) -> Optional[ComponentMetadata]:
        required_keys = {
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.weight",
        }

        return (
            ComponentMetadata(
                display_name="CLIP Text Encoder",
                input_space="SDXL",
                output_space="SDXL",
            )
            if all(key in state_dict for key in required_keys)
            else None
        )

    @override
    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None):
        print("Loading SDXL TextEncoder")
        start = time.time()

        text_encoder = self.model

        text_encoder_state_dict = {
            key: state_dict[key]
            for key in state_dict
            if key.startswith("conditioner.embedders.0.transformer.")
        }
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
            unexpected_keys = load_model_dict_into_meta(
                text_encoder, text_model_dict, dtype=torch_dtype
            )
            if text_encoder._keys_to_ignore_on_load_unexpected is not None:
                for pat in text_encoder._keys_to_ignore_on_load_unexpected:
                    unexpected_keys = [
                        k for k in unexpected_keys if re.search(pat, k) is None
                    ]

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint were not used when initializing {text_encoder.__class__.__name__}: \n {[', '.join(unexpected_keys)]}"
                )
        else:
            if not (
                hasattr(text_encoder, "embeddings")
                and hasattr(text_encoder.embeddings.position_ids)
            ):
                text_model_dict.pop("text_model.embeddings.position_ids", None)


            text_encoder.load_state_dict(text_model_dict)

        if device is not None:
            text_encoder.to(device=device)
        # text_encoder.to(torch.float16)

        # text_encoder.to_empty(device=torch.device("cuda"))

        print(f"TextEncoder loaded in {time.time() - start} seconds")
