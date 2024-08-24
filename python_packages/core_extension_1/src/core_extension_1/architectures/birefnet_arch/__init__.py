import os
import json
import time
from typing import Any, Optional

from typing_extensions import override
from gen_server import Architecture, StateDict, TorchDevice, ComponentMetadata
import torch

from .birefnet import BiRefNet as BiRefNetModel

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


class BiRefNet(Architecture[BiRefNetModel]):
    def __init__(self, **ignored: Any):
        super().__init__()
        with open(config_path, "r") as file:
            config = json.load(file)
            model = BiRefNetModel(**config)
            self._model = model
            self._config = config

        self._display_name = "BiRefNet"
        self._input_space = "NET"
        self._output_space = "NET"

    @classmethod
    def detect(
        cls,
        state_dict: Optional[StateDict] = None,
        **ignored: Any,
    ) -> Optional[ComponentMetadata]:
        required_keys = {}

        return (
            ComponentMetadata(
                display_name="BiRefNet",
                input_space="NET",
                output_space="NET",
            )
            if all(key in state_dict for key in required_keys)
            else None
        )

    @override
    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None):
        print("Loading BiRefNet")
        start = time.time()

        self.model.load_state_dict(state_dict)

        if device is not None:
            self.model.to(device=device)
        self.model.to(torch.bfloat16)

        print(f"BiRefNet loaded in {time.time() - start} seconds")
