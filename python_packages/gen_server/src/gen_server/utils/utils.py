import re
from typing import Dict, Any
from pydantic import BaseModel



def flatten_architectures(architectures):
    flat_architectures = {}
    for arch_id, architecture in architectures.items():
        if isinstance(architecture, list):
            for arch in architecture:
                flat_architectures[f"{arch_id}:{arch.__name__}"] = architecture
        else:
            flat_architectures[arch_id] = architecture

    return flat_architectures


def to_snake_case(value):
    """
    Convert CamelCase to snake_case
    """
    pattern = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
    return pattern.sub("_", value).lower()


def serialize_config(config: BaseModel) -> Dict[str, Any]:
    """
    Serialize a Pydantic model (like RunCommandConfig) into a dictionary.
    This function handles nested Pydantic models and converts them to dictionaries as well.
    """
    def serialize(obj):
        if isinstance(obj, BaseModel):
            return {k: serialize(v) for k, v in obj.dict().items()}
        elif isinstance(obj, list):
            return [serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        else:
            return obj

    return serialize(config)
