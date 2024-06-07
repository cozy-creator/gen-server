from gen_server.extension_loader import load_components
from .arch_definition import ArchDefinition

ARCHITECTURES = load_components('comfy_creator.architectures', expected_type=ArchDefinition)
"""
Global class containing all architecture definitions
"""
