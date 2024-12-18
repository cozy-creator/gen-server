import json
from typing import Dict, Any
import os
from ..config import get_config

# model_config_path = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), "model_config.json"
# )


# TO DO: this should instead be more like, better default parameters for each
# model.

DEFAULTS = {
    "diffuser_class_defaults": {
      "StableDiffusionXLPipeline": {
        "scheduler": "",
        "guidance_scale": 3.5,
        "num_inference_steps": 25
      },
      "StableDiffusionPipeline": {
        "scheduler": "",
        "guidance_scale": 3.5,
        "num_inference_steps": 25
      },
      "StableDiffusion3Pipeline": {
        "scheduler": "",
        "guidance_scale": 3.5,
        "num_inference_steps": 28
      }
    },
    "global_default": {
      "scheduler": "",
      "guidance_scale": 3.5,
      "num_inference_steps": 25
    }
}


class ModelConfigManager:
    def __init__(self):
        # self.config_path = config_path
        self.config: Dict[str, Any] = DEFAULTS

    def get_model_config(self, pipeline_name: str, class_name: str) -> Dict[str, Any]:
        model_config = get_config().pipeline_defs.get(pipeline_name, {}).get("default_args", {})
        print(f"model_config: {model_config}")

        # Get global default settings
        global_config = self.config["global_default"]

        # Get class default settings
        class_config = self.config["diffuser_class_defaults"].get(class_name, {})

        # Merge configurations, with the order of precedence being:
        # model_config > class_config > global_config
        final_config = {
            **global_config,
            **class_config,
            **model_config,
        }

        print(final_config)

        return final_config

    # def get_scheduler(self, pipeline_name: str, class_name: str) -> str:
    #     return self.get_model_config(pipeline_name, class_name)["scheduler"]

    # def get_default_positive_prompt(self, pipeline_name: str, class_name: str) -> str:
    #     return self.get_model_config(pipeline_name, class_name)["default_positive_prompt"]

    # def get_default_negative_prompt(self, pipeline_name: str, class_name: str) -> str:
    #     return self.get_model_config(pipeline_name, class_name)["default_negative_prompt"]

    # def get_guidance_scale(self, pipeline_name: str, class_name: str) -> float:
    #     return self.get_model_config(pipeline_name, class_name)["guidance_scale"]

    # def get_num_inference_steps(self, pipeline_name: str, class_name: str) -> int:
    #     return self.get_model_config(pipeline_name, class_name)["num_inference_steps"]
