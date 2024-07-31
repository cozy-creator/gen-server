import json
from typing import Dict, Any
import os

model_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_config.json")

class ModelConfigManager:
    def __init__(self, config_path: str = model_config_path):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

    def get_model_config(self, repo_id: str, class_name: str) -> Dict[str, Any]:
        if repo_id in self.config['models']:
            return self.config['models'][repo_id]
        elif class_name in self.config['class_defaults']:
            return self.config['class_defaults'][class_name]
        else:
            return self.config['global_default']

    def get_scheduler(self, repo_id: str, class_name: str) -> str:
        return self.get_model_config(repo_id, class_name)['scheduler']

    def get_default_positive_prompt(self, repo_id: str, class_name: str) -> str:
        return self.get_model_config(repo_id, class_name)['default_positive_prompt']

    def get_default_negative_prompt(self, repo_id: str, class_name: str) -> str:
        return self.get_model_config(repo_id, class_name)['default_negative_prompt']

    def get_guidance_scale(self, repo_id: str, class_name: str) -> float:
        return self.get_model_config(repo_id, class_name)['guidance_scale']

    def get_num_inference_steps(self, repo_id: str, class_name: str) -> int:
        return self.get_model_config(repo_id, class_name)['num_inference_steps']