from typing import Callable
from diffusers import AutoencoderKL
from spandrel_core import Architecture, StateDict
from spandrel_core.util import KeyCondition
import json
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
import time
from paths import folders


class SD15VAEArch(Architecture[AutoencoderKL]):
    def __init__(
            self,
    ) -> None:
        super().__init__(
            id="SD15VAE",
            name="VAE",
            detect=KeyCondition.has_all(
                "first_stage_model.encoder.conv_in.bias",
            ),
        )

    def load(self, state_dict: StateDict) -> AutoencoderKL:
        print("Loading SD1.5 VAE")
        start = time.time()
        config = json.load(open(f"{folders['vae']}/sd15_vae_config.json"))
        vae = AutoencoderKL(**config)

        vae_state_dict = {key: state_dict[key] for key in state_dict if key.startswith("first_stage_model.")}

        new_vae_state_dict = convert_ldm_vae_checkpoint(vae_state_dict, config=config)
        vae.load_state_dict(new_vae_state_dict)
        print(f"VAE state dict loaded in {time.time() - start} seconds")
        return {
            "vae": vae,
            "lineage": "SD1.5"
        }



