import json
import time
from typing import Dict, Optional, Any
from gen_server import (
    Architecture,
    StateDict,
    TorchDevice,
    CheckpointMetadata,
    ComponentMetadata,
)
from transformers import CLIPTextModelWithProjection, CLIPTextConfig
from diffusers.utils.import_utils import is_accelerate_available
from diffusers.models.model_loading_utils import load_model_dict_into_meta
from diffusers.loaders.single_file_utils import convert_open_clip_checkpoint
import re
import os
import torch
import logging
from contextlib import nullcontext
from gen_server.utils.device import get_torch_device

logger = logging.getLogger(__name__)

if is_accelerate_available():
    from accelerate import init_empty_weights


class TextEncoder2(Architecture[CLIPTextModelWithProjection]):
    """
    Architecture definition for the Text Encoder 2 (CLIP-based).
    """

    @staticmethod
    def _determine_type(metadata: dict[str, Any]) -> tuple[ComponentMetadata, str]:
        architecture = metadata.get("modelspec.architecture", "")

        if architecture == "stable-diffusion-v3-medium":
            result: ComponentMetadata = {
                "display_name": "SD3 Text Encoder 2",
                "input_space": "SD3",
                "output_space": "SD3",
            }
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "config_sd3.json"
            )
        # elif architecture == "stable-diffusion-xl-v1-base":
        else:
            result: ComponentMetadata = {
                "display_name": "SDXL Text Encoder 2",
                "input_space": "SDXL",
                "output_space": "SDXL",
            }
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "config_sdxl.json"
            )

        return result, config_path

    def __init__(self, metadata: dict[str, Any]):
        result, config_path = self._determine_type(metadata)
        self._display_name = result["display_name"]
        self._input_space = result["input_space"]
        self._output_space = result["output_space"]

        with open(config_path, "r") as file:
            # Create diffusers class
            config = json.load(file)
        text_encoder_config = CLIPTextConfig.from_dict(config)
        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            text_encoder = CLIPTextModelWithProjection(text_encoder_config)

        self._model = text_encoder
        self._config = text_encoder_config

    @classmethod
    def detect(
        cls,
        state_dict: StateDict,
        metadata: dict[str, Any],
    ) -> Optional[ComponentMetadata]:
        required_keys = {
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
            "text_encoders.clip_g.transformer.text_model.embeddings.position_embedding.weight",
            # "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight"
        }
        """
        Detects whether the given state dictionary matches the SD3 Text Encoder 2 architecture.
        """
        if any(key in state_dict for key in required_keys):
            component_metadata, _ = cls._determine_type(metadata)
            return component_metadata

        return None

    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None):
        """
        Loads the SDXL Text Encoder 2 model from the given state dictionary.
        """
        print("Loading SDXL Text Encoder 2")
        text_model = self.model
        text_encoder_state_dict = {
            key: state_dict[key]
            for key in state_dict
            if key.startswith("conditioner.embedders.1.model.")
            or key.startswith("text_encoders.clip_g.")
        }
        text_model_dict = {}

        if any(
            key.startswith("text_encoders.clip_g.") for key in text_encoder_state_dict
        ):
            prefix = "text_encoders.clip_g."
        elif any(
            key.startswith("conditioner.embedders.1.model.")
            for key in text_encoder_state_dict
        ):
            prefix = "conditioner.embedders.1.model."
        else:
            prefix = None

        text_model_dict = convert_open_clip_checkpoint(
            text_model=text_model, checkpoint=text_encoder_state_dict, prefix=prefix
        )

        print(text_model_dict.keys())

        if is_accelerate_available():
            print("Using accelerate")
            # torch_dtype = next(text_model.parameters()).dtype
            unexpected_keys = load_model_dict_into_meta(text_model, text_model_dict)
            if text_model._keys_to_ignore_on_load_unexpected is not None:
                for pat in text_model._keys_to_ignore_on_load_unexpected:
                    unexpected_keys = [
                        k for k in unexpected_keys if re.search(pat, k) is None
                    ]

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint were not used when initializing {text_model.__class__.__name__}: \n {[', '.join(unexpected_keys)]}"
                )

        else:
            text_model.load_state_dict(text_model_dict)
            # text_model.to(torch.float16)
        text_model.to_empty(device=get_torch_device())

        # text_model.to(device=get_tensor_device()
