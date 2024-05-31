import torch
from typing import Dict
from model_registry._helpers.architectures import ModelArchitecture, StateDict


class FirstArchitecture(torch.nn.Module):
    """
    The PyTorch module for your custom architecture.
    (Define the model structure, layers, etc.)
    """
    def __init__(self, **kwargs):
        super().__init__()
        # ... first model definition here

    def forward(self, x, **kwargs):
        # ... first model's forward pass
        pass


class FirstArchitectureArch(ModelArchitecture):
    def __init__(self):
        super().__init__(
            id="my_architecture",
            required_keys=[
                # ... specify the required keys for the architecture
            ],
            key_converter=lambda key: key,
            decomposer=self._decompose_my_architecture,
            
        )

    def _decompose_my_architecture(self, state_dict: StateDict) -> Dict[str, StateDict]:
        # ... state dictionary decomposition logic 
        pass