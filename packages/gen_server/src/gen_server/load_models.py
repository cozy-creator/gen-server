import os
from typing import List

import torch

from gen_server import (
    StateDict,
    Checkpoint,
)
from gen_server.globals import comfy_config, PRETRAINED_MODELS

ARCH_KEYS = {
    "sd1.5": [
        "model.diffusion_model.input_blocks.0.0.weight",
        "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
    ],
    "sdxl": [
        "model.diffusion_model.input_blocks.0.0.weight",
        "conditioner.embedders.1.model.token_embedding.weight",
    ],
    # "sd1": [
    #     "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
    # ],
}


def load_models():
    for model_dir in comfy_config.models_dirs:
        for dirpath, _dirnames, filenames in os.walk(model_dir):
            for filename in filenames:
                model_file = os.path.join(dirpath, filename)
                if not os.path.isfile(model_file):
                    print(f"Error: File '{model_file}' not found.")
                    continue

                try:
                    # We need to get the device based on the one available on the machine, not just "cuda"
                    model = torch.load(str(model_file), map_location="cuda")
                    for name, keys in ARCH_KEYS.items():
                        if has_all_keys(keys, model):
                            models = PRETRAINED_MODELS.get("")
                            if models is None:
                                checkpoint = Checkpoint()
                                checkpoint.components.append(model)
                                PRETRAINED_MODELS[name] = checkpoint
                            else:
                                PRETRAINED_MODELS.update({name: models})
                            break
                except Exception as e:
                    print(
                        f"Error: Unexpected error while loading model from file '{model_file}': {e}"
                    )
                    continue


def has_all_keys(keys: List[str], state_dict: StateDict) -> bool:
    """
    Detects if the given state dictionary matches this architecture.
    """
    for key in keys:
        if key not in state_dict:
            return False
    return True
