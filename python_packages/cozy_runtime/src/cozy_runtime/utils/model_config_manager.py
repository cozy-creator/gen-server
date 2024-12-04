import json
from typing import Dict, Any
import os
from ..config import get_config

model_config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model_config.json"
)


# TO DO: this should instead be more like, better default parameters for each
# model.

DEFAULTS = {
    "diffuser_class_defaults": {
      "StableDiffusionXLPipeline": {
        "scheduler": "",
        "default_positive_prompt": "",
        "default_negative_prompt": "",
        "guidance_scale": 7.0,
        "num_inference_steps": 20
      },
      "StableDiffusionPipeline": {
        "scheduler": "",
        "default_positive_prompt": "",
        "default_negative_prompt": "",
        "guidance_scale": 7.0,
        "num_inference_steps": 25
      },
      "StableDiffusion3Pipeline": {
        "scheduler": "",
        "default_positive_prompt": "",
        "default_negative_prompt": "",
        "guidance_scale": 7.0,
        "num_inference_steps": 28
      }
    },
    "global_default": {
      "scheduler": "",
      "default_positive_prompt": "",
      "default_negative_prompt": "",
      "guidance_scale": 7.0,
      "num_inference_steps": 25
    }
}


class ModelConfigManager:
    def __init__(self, config_path: str = model_config_path):
        self.config_path = config_path
        self.config: Dict[str, Any] = DEFAULTS
        # self.load_config()

    # def load_config(self):
    #     if os.path.exists(self.config_path):
    #         with open(self.config_path, "r") as f:
    #             self.config = json.load(f)
    #     else:
    #         raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

    def get_model_config(self, repo_id: str, class_name: str) -> Dict[str, Any]:
        model_config = get_config().pipeline_defs.get(repo_id, {}).get("default_args", {})
        print(f"model_config: {model_config}")

        # Get global default settings
        global_config = self.config["global_default"]

        # Get class default settings
        class_config = self.config["diffuser_class_defaults"].get(class_name, {})

        # Get category settings
        # model_config = self.config["repo_defaults"].get(repo_id, {})
        # category = model_config.get("category", None)
        # category_config = (
        #     self.config["cozy_categories"].get(category, {}) if category else {}
        # )

        # Merge configurations, with the order of precedence being:
        # model_config > category_config > class_config > global_config
        final_config = {
            **global_config,
            **class_config,
            # **category_config,
            **model_config,
        }

        print(final_config)

        return final_config

    def get_scheduler(self, repo_id: str, class_name: str) -> str:
        return self.get_model_config(repo_id, class_name)["scheduler"]

    def get_default_positive_prompt(self, repo_id: str, class_name: str) -> str:
        return self.get_model_config(repo_id, class_name)["default_positive_prompt"]

    def get_default_negative_prompt(self, repo_id: str, class_name: str) -> str:
        return self.get_model_config(repo_id, class_name)["default_negative_prompt"]

    def get_guidance_scale(self, repo_id: str, class_name: str) -> float:
        return self.get_model_config(repo_id, class_name)["guidance_scale"]

    def get_num_inference_steps(self, repo_id: str, class_name: str) -> int:
        return self.get_model_config(repo_id, class_name)["num_inference_steps"]
