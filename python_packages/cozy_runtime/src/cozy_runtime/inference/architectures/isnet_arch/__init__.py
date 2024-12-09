import os
import json
import time
from typing import Any, Optional

from typing_extensions import override
from cozy_runtime import Architecture, StateDict, TorchDevice, ComponentMetadata
import torch

from .isnet import ISNetDIS as ISNetDISModel

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


class ISNetDIS(Architecture[ISNetDISModel]):
    def __init__(self, **ignored: Any):
        super().__init__()
        with open(config_path, "r") as file:
            config = json.load(file)
            model = ISNetDISModel(**config)
            self._model = model
            self._config = config

        self._display_name = "ISNetDIS"
        self._input_space = "DIS"
        self._output_space = "DIS"

    @classmethod
    def detect(
        cls,
        state_dict: Optional[StateDict] = None,
        **ignored: Any,
    ) -> Optional[ComponentMetadata]:
        required_keys = {
            "is_net"
        }  # Ensure to change this to its actual keys. This is just a random key to prevent it from being detected by other models

        return (
            ComponentMetadata(
                display_name="ISNetDIS",
                input_space="DIS",
                output_space="DIS",
            )
            if all(key in state_dict for key in required_keys)
            else None
        )

    @override
    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None):
        print("Loading ISNetDIS")
        start = time.time()

        self.model.load_state_dict(state_dict)

        if device is not None:
            self.model.to(device=device)
        self.model.to(torch.bfloat16)

        print(f"ISNetDIS loaded in {time.time() - start} seconds")
