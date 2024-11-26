import json
import time
from typing import Optional, Any
from cozy_runtime import Architecture, StateDict, TorchDevice, ComponentMetadata
from transformers import CLIPTextModelWithProjection, CLIPTextConfig, CLIPTextModel
from diffusers.utils.import_utils import is_accelerate_available
from diffusers.models.model_loading_utils import load_model_dict_into_meta
import re
import os
import torch
import logging
from contextlib import nullcontext

from cozy_runtime.utils.device import get_torch_device

logger = logging.getLogger(__name__)

if is_accelerate_available():
    from accelerate import init_empty_weights

LDM_CLIP_PREFIX_TO_REMOVE = [
    "cond_stage_model.transformer.",
    "conditioner.embedders.0.transformer.",
    "text_encoders.clip_l.transformer.",
]

config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config_text_encoder_1.json"
)


class FluxTextEncoder1(Architecture[CLIPTextModel]):
    """
    Architecture definition for the SD3 Text Encoder 1 (CLIP-based).
    """

    def __init__(self, **ignored: Any):
        with open(config_path, "r") as file:
            config = json.load(file)
            text_encoder_config = CLIPTextConfig.from_dict(config)
        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            text_encoder = CLIPTextModel(text_encoder_config)

        self._model = text_encoder
        self._config = config

        self._display_name = "CLIP Text Encoder"
        self._input_space = "SD3"
        self._output_space = "SD3"

    @classmethod
    def detect(  # type: ignore
        cls, state_dict: StateDict, **ignored: Any
    ) -> Optional[ComponentMetadata]:
        """
        Detects whether the given state dictionary matches the SD3 Text Encoder 1 architecture.
        """
        state_key = "text_encoders.clip_l.transformer.text_model.embeddings.position_embedding.weight"

        return (
            ComponentMetadata(
                display_name="CLIP Text Encoder",
                input_space="SD3",
                output_space="SD3",
            )
            if state_key in state_dict
            else None
        )

    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None):
        """
        Loads the Flux Text Encoder 1 model from the given state dictionary.
        """
        print("Loading SD3 Text Encoder 1")
        start = time.time()

        text_encoder = self._model
        text_encoder_state_dict = {
            key: state_dict[key]
            for key in state_dict
            if key.startswith("text_encoders.clip_l.")
        }

        remove_prefixes = LDM_CLIP_PREFIX_TO_REMOVE
        keys = list(text_encoder_state_dict.keys())
        text_model_dict = {}

        for key in keys:
            for prefix in remove_prefixes:
                if key.startswith(prefix):
                    diffusers_key = key.replace(prefix, "")
                    text_model_dict[diffusers_key] = text_encoder_state_dict[key]

        # print(text_model_dict.keys())

        if is_accelerate_available():
            print("Using accelerate")
            torch_dtype = next(text_encoder.parameters()).dtype
            unexpected_keys = load_model_dict_into_meta(
                text_encoder, text_model_dict, dtype=torch.bfloat16
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
            text_encoder.to(torch.float16)

        # text_encoder.to_empty(device=get_torch_device())
        # text_encoder.to(torch.float16)
        # text_encoder.to("cuda")
