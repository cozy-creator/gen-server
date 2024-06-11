from typing import Dict, Tuple, Union
from gen_server.types import WidgetDefinition


class StringInput(WidgetDefinition):
    """String input widget"""
    def __init__(self, multiline=False, max_length=None):
        self.multiline = multiline
        self.max_length = max_length


class NumberInput(WidgetDefinition):
    """Number input widget"""
    def __init__(self, step=1, default=None, max=None, min=None):
        self.step = step
        self.default = default
        self.max = max
        self.min = min


class BooleanInput(WidgetDefinition):
    """Boolean input widget"""
    def __init__(self, default=False):
        self.default = default
