import os
from gen_server import ArchDefinition, StateDict, ModelWrapper
import json
import time
from transformers import CLIPTextModel, CLIPTextConfig

LDM_CLIP_PREFIX_TO_REMOVE = ["cond_stage_model.transformer.", "conditioner.embedders.0.transformer."]

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


class SD1TextEncoderArch(ArchDefinition[CLIPTextModel]):
    """
    The CLIP text-encoder used for the Stable Diffusion 1 pipeline
    """
    
    @classmethod
    def detect(self, state_dict: StateDict) -> bool:
        required_keys = {
            "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
            # "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight",
        }
        
        return all(key in state_dict for key in required_keys)
    
    @classmethod
    def load(self, state_dict: StateDict) -> ModelWrapper[CLIPTextModel]:
        print("Loading SD1.5 TextEncoder")
        start = time.time()
        with open(config_path, 'r') as file:
            config = json.load(file)

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

        return ModelWrapper(text_encoder)
    