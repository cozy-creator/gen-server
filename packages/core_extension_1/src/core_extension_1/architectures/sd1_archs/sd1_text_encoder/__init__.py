import os
import json
import time
from typing import Optional, Any
from typing_extensions import override
from gen_server import Architecture, StateDict, TorchDevice, ComponentMetadata
from transformers import CLIPTextModel, CLIPTextConfig
import torch

LDM_CLIP_PREFIX_TO_REMOVE = [
    "cond_stage_model.transformer.",
    "conditioner.embedders.0.transformer.",
]

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


class SD1TextEncoder(Architecture[CLIPTextModel]):
    """
    The CLIP text-encoder used for the Stable Diffusion 1 pipeline
    """

    def __init__(self):
        with open(config_path, "r") as file:
            config = json.load(file)
            text_encoder_config = CLIPTextConfig.from_dict(config)
            text_encoder = CLIPTextModel(text_encoder_config)

        self._model = text_encoder
        self._config = text_encoder_config
        self._display_name = "CLIP Text Encoder"
        self._input_space = "SD1"
        self._output_space = "SD1"

    @classmethod
    def detect(
        cls,
        state_dict: StateDict,
        metadata: dict[str, Any],
    ) -> Optional[ComponentMetadata]:
        required_keys = {
            "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
            # "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight",
        }

        return (
            ComponentMetadata(
                display_name="CLIP Text Encoder",
                input_space="SD1",
                output_space="SD1",
            )
            if all(key in state_dict for key in required_keys)
            else None
        )

    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None):
        print("Loading SD1.5 TextEncoder")
        start = time.time()

        text_encoder = self.model

        text_encoder_state_dict = {
            key: state_dict[key]
            for key in state_dict
            if key.startswith("cond_stage_model.transformer.")
        }
        remove_prefixes = LDM_CLIP_PREFIX_TO_REMOVE
        keys = list(text_encoder_state_dict.keys())
        text_model_dict = {}

        for key in keys:
            for prefix in remove_prefixes:
                if key.startswith(prefix):
                    diffusers_key = key.replace(prefix, "")
                    text_model_dict[diffusers_key] = text_encoder_state_dict[key]

        if not (
            hasattr(text_encoder, "embeddings")
            and hasattr(text_encoder.embeddings, "position_ids")
        ):
            text_model_dict.pop("text_model.embeddings.position_ids", None)

        text_encoder.load_state_dict(text_model_dict)

        if device is not None:
            text_encoder.to(device=device) # type: ignore
        text_encoder.to(torch.bfloat16) # type: ignore

        print(f"TextEncoder loaded in {time.time() - start} seconds")

if __name__ == "__main__":
    passes = issubclass(SD1TextEncoder, Architecture)
    print(f"Passes: {passes}")
    