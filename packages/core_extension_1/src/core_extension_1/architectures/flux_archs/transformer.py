import json
import time
from typing import Optional, Any
from gen_server import Architecture, StateDict, TorchDevice, ComponentMetadata
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.loaders.single_file_utils import (
    convert_flux_transformer_checkpoint_to_diffusers,
)
from diffusers.utils.import_utils import is_accelerate_available
from diffusers.models.model_loading_utils import load_model_dict_into_meta
import torch
import logging
import re
import os
from contextlib import nullcontext

logger = logging.getLogger(__name__)

config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config_transformer.json"
)


if is_accelerate_available():
    from accelerate import init_empty_weights


class FluxTransformer(Architecture[FluxTransformer2DModel]):
    """
    Architecture definition for the Flux Transformer model.
    """

    def __init__(self, **ignored: Any):
        with open(config_path, "r") as file:
            # Create diffusers class
            config = json.load(file)
        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            model = FluxTransformer2DModel(**config)

        self._model = model
        self._config = config

        self._display_name = "Flux Transformer"
        self._input_space = "Flux"
        self._output_space = "Flux"

    @classmethod
    def detect( # type: ignore
        cls, state_dict: StateDict, **ignored: Any
    ) -> Optional[ComponentMetadata]:
        """
        Detects whether the given state dictionary matches the Flux transformer architecture.
        """
        required_keys = {
            "model.diffusion_model.joint_blocks.0.context_block.attn.proj.bias",
            "model.diffusion_model.joint_blocks.0.context_block.attn.proj.weight",
        }

        return (
            ComponentMetadata(
                display_name="Flux Transformer",
                input_space="Flux",
                output_space="Flux",
            )
            if all(key in state_dict for key in required_keys)
            else None
        )

    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None):
        """
        Loads the Flux Transformer model from the given state dictionary.
        """
        print("Loading Flux Transformer")
        start = time.time()

        transformer = self._model
        # transformer_state_dict = {
        #     key: state_dict[key]
        #     for key in state_dict
        #     if key.startswith("model.diffusion_model.")
        # }
        new_transformer_state_dict = convert_flux_transformer_checkpoint_to_diffusers(
            state_dict, config=self._config
        )

        if is_accelerate_available():
            print("Using accelerate")
            unexpected_keys = load_model_dict_into_meta(
                transformer, new_transformer_state_dict, dtype=torch.bfloat16
            )
            if transformer._keys_to_ignore_on_load_unexpected is not None:
                for pat in transformer._keys_to_ignore_on_load_unexpected:
                    unexpected_keys = [
                        k for k in unexpected_keys if re.search(pat, k) is None
                    ]

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint were not used when initializing {transformer.__name__}: \n {[', '.join(unexpected_keys)]}"
                )
        else:
            transformer.load_state_dict(new_transformer_state_dict)
            transformer.to(torch.float16)


        print(f"Transformer loaded in {time.time() - start:.2f} seconds")
        
