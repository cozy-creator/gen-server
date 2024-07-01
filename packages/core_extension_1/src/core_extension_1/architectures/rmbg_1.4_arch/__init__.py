import os
import json
import time
from typing import Any, Optional

from typing_extensions import override
from gen_server import Architecture, StateDict, TorchDevice, ComponentMetadata
from transformers import CLIPTextModel, CLIPTextConfig
import torch

from .briarmbg.briarmbg import BriaRMBG

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


class RMBGArch(Architecture[CLIPTextModel]):
    def __init__(self):
        super().__init__()
        with open(config_path, "r") as file:
            config = json.load(file)
            model = BriaRMBG(**config)
            self._model = model
            self._config = config

        self._display_name = "RMBG 1.4"
        self._input_space = "RMBG"
        self._output_space = "RMBG"

    @classmethod
    def detect(
        cls,
        state_dict: Optional[StateDict] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[ComponentMetadata]:
        required_keys = {
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.weight",
        }

        return (
            ComponentMetadata(
                display_name="RMBG 1.4",
                input_space="RMBG",
                output_space="RMBG",
            )
            if all(key in state_dict for key in required_keys)
            else None
        )

    @override
    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None):
        print("Loading RMBG")
        start = time.time()

        self.model.load_state_dict(state_dict)

        if device is not None:
            self.model.to(device=device)
        self.model.to(torch.bfloat16)

        print(f"RMBG loaded in {time.time() - start} seconds")
