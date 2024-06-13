from enum import Enum
from typing import Dict, Union, Protocol

from gen_server.types_1.types_1 import Serializable


class WidgetDefinition(Serializable):
    """Base class for widget definitions"""

    def serialize(self):
        """
        Serialize the object into a dictionary
        :return: serialized object
        """
        result = {}
        for attr in self.__dir__():
            if not attr.startswith("_") and not callable(self.__getattribute__(attr)):
                result[attr] = self.__getattribute__(attr)
        return result


class TextInput(WidgetDefinition):
    """Text input widget"""

    def __init__(self, default=None, value= None, max_length=None):
        self.max_length = max_length


class StringInput(WidgetDefinition):
    """String input widget"""

    def __init__(self, max_length=None):
        self.max_length = max_length


class EnumInput(WidgetDefinition):
    """Enum input widget"""

    def __init__(self, default=None, options=None):
        if options is None:
            options = []
        self.options = options
        if default is not None and default not in options:
            raise ValueError(f"Invalid default value: {default}")
        self.default = default


class IntInput(WidgetDefinition):
    """Integer input widget"""

    def __init__(self, step=1, default=None, max=None, min=None):
        self.min = min
        self.max = max
        self.step = step
        self.default = default


class FloatInput(WidgetDefinition):
    """Float input widget"""

    def __init__(self, step=1, default=None, max=None, min=None):
        self.min = min
        self.max = max
        self.step = step
        self.default = default


class BooleanInput(WidgetDefinition):
    """Boolean input widget"""

    def __init__(self, default=False):
        self.default = default
