import re


def flatten_architectures(architectures):
    flat_architectures = {}
    for arch_id, architecture in architectures.items():
        if isinstance(architecture, list):
            for architecture in architecture:
                flat_architectures[f"{arch_id}:{architecture.__name__}"] = architecture
        else:
            flat_architectures[arch_id] = architecture

    return flat_architectures


def to_snake_case(value):
    """
    Convert CamelCase to snake_case
    """
    pattern = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
    return pattern.sub("_", value).lower()
