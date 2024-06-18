from __future__ import annotations
from .registry import ArchRegistry, ArchSupport


MAIN_REGISTRY = ArchRegistry(namespace="comfy_creator")
"""
A global class containing all comfy-creator architectures discovered in the Python environment
"""
