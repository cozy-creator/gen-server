import os
import json
import time
from typing import Any, Optional

from typing_extensions import override
from gen_server import Architecture, StateDict, TorchDevice, ComponentMetadata
import torch

from .briarmbg import BriaRMBG as BriaRMBGModel

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


class BriaRMBG(Architecture[BriaRMBGModel]):
    def __init__(self, **ignored: Any):
        super().__init__()
        with open(config_path, "r") as file:
            config = json.load(file)
            model = BriaRMBGModel(**config)
            self._model = model
            self._config = config

        self._display_name = "BriaRMBG 1.4"
        self._input_space = "RMBG"
        self._output_space = "RMBG"

    @classmethod
    def detect(
        cls,
        state_dict: Optional[StateDict] = None,
        **ignored: Any,
    ) -> Optional[ComponentMetadata]:
        required_keys = {
            "stage1.rebnconvin.conv_s1.weight",
            "stage1.rebnconvin.conv_s1.bias",
        }

        return (
            ComponentMetadata(
                display_name="BriaRMBG 1.4",
                input_space="RMBG",
                output_space="RMBG",
            )
            if all(key in state_dict for key in required_keys)
            else None
        )

    @override
    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None):
        print("Loading BriaRMBG")
        start = time.time()

        self.model.load_state_dict(state_dict)

        if device is not None:
            self.model.to(device=device)
        self.model.to(torch.bfloat16)

        print(f"BriaRMBG loaded in {time.time() - start} seconds")
