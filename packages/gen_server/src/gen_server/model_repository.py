import importlib
import json
import os
from typing import List

from packages.gen_server.src.gen_server import ModelConstraint, Architecture
from packages.gen_server.src.gen_server.globals import comfy_config


class ModelRepository:
    def __init__(self, index_file="models.index"):
        self.models = []
        self.models_by_space = {}
        self.index_file = index_file

        if os.path.exists(self.index_file):
            self.load_models_from_index()
        else:
            self.load_models(comfy_config)
            self.index_models()

    def load_models_from_index(self):
        try:
            with open(self.index_file, "r") as f:
                models_data = json.load(f)
                self.models = [
                    ModelConstraint(**model_data) for model_data in models_data
                ]
                self.group_models()
        except FileNotFoundError:
            print(f"Error: Index file '{self.index_file}' not found.")
        except json.JSONDecodeError:
            print(f"Error: Index file '{self.index_file}' is not a valid JSON file.")
        except Exception as e:
            print(f"Error: Unexpected error while loading index file: {e}")

    def index_models(self):
        try:
            with open(self.index_file, "w") as f:
                models_data = [model.__dict__ for model in self.models]
                json.dump(models_data, f)
        except Exception as e:
            print(f"Error: Unexpected error while saving index file: {e}")

    def load_models(self, comfy_config):
        for model_dir in comfy_config.models_dirs:
            for dirpath, dirnames, filenames in os.walk(model_dir):
                for filename in filenames:
                    if filename.endswith(".py"):
                        module_name = os.path.splitext(filename)[0]
                        module_file = os.path.join(dirpath, filename)

                        if not os.path.isfile(module_file):
                            print(f"Error: File '{module_file}' not found.")
                            continue

                        try:
                            module_spec = importlib.util.spec_from_file_location(
                                module_name, module_file
                            )
                            module = importlib.util.module_from_spec(module_spec)
                            module_spec.loader.exec_module(module)
                        except Exception as e:
                            print(
                                f"Error: Unexpected error while importing module '{module_name}': {e}"
                            )
                            continue

                        model = None

                        for name, item in module.__dict__.items():
                            if isinstance(item, type) and issubclass(
                                item, Architecture
                            ):
                                model = item
                                break

                        if model is None:
                            print(
                                f"Error: No model class found in module '{module_name}'."
                            )
                            continue

                        if any(model == m for m in self.models):
                            print(f"Error: Duplicate model found: {model}")
                            continue

                        self.models.append(model)

        self.group_models()

    def group_models(self):
        self.models_by_space = {}
        for model in self.models:
            input_space = model.input_space
            if input_space is None:
                continue
            if input_space not in self.models_by_space:
                self.models_by_space[input_space] = []
            self.models_by_space[input_space].append(model)

    def get_models_by_space(self, model_type: str) -> List[ModelConstraint]:
        return self.models_by_space.get(model_type, [])
