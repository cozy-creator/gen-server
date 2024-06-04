from spandrel import Architecture, StateDict
from spandrel.util import KeyCondition
import json
import time
from transformers import CLIPTextModel, CLIPTextConfig
from paths import folders


LDM_CLIP_PREFIX_TO_REMOVE = ["cond_stage_model.transformer.", "conditioner.embedders.0.transformer."]



class SD15TextEncoderArch(Architecture[CLIPTextModel]):
    def __init__(
            self,
    ) -> None:
        super().__init__(
            id="SD15TextEncoder",
            name="TextEncoder",
            detect=KeyCondition.has_any(
                "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
                "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight",
            ),
        )

    def load(self, state_dict: StateDict) -> CLIPTextModel:
        print("Loading SD1.5 TextEncoder")
        start = time.time()
        config = json.load(open(f"{folders['text_encoder']}/sd15_text_config.json"))

        text_encoder_config = CLIPTextConfig.from_dict(config)
        text_encoder = CLIPTextModel(text_encoder_config)

        text_encoder_state_dict = {key: state_dict[key] for key in state_dict if key.startswith("cond_stage_model.transformer.")}
        remove_prefixes = LDM_CLIP_PREFIX_TO_REMOVE
        keys = list(text_encoder_state_dict.keys())
        text_model_dict = {}

        for key in keys:
            for prefix in remove_prefixes:
                if key.startswith(prefix):
                    diffusers_key = key.replace(prefix, "")
                    text_model_dict[diffusers_key] = text_encoder_state_dict[key]

        if not (hasattr(text_encoder, "embeddings") and hasattr(text_encoder.embeddings.position_ids)):
            text_model_dict.pop("text_model.embeddings.position_ids", None)

        text_encoder.load_state_dict(text_model_dict)

        print(f"TextEncoder loaded in {time.time() - start} seconds")

        return {
            "text_encoder": text_encoder,
            "lineage": "SD1.5"
        }


