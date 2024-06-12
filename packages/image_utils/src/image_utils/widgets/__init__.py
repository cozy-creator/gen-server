from abc import ABC
from enum import Enum
from typing import Dict, Union, Protocol


# from gen_server.types import WidgetDefinition


class WidgetType(Enum):
    """Widget type enum"""

    TEXT = "text"
    """Text widget"""

    STRING = "string"
    """String input widget"""

    INT = "int"
    """Integer input widget"""

    ENUM = "enum"
    """Enum input widget"""

    FLOAT = "float"
    """Float input widget"""

    BOOLEAN = "boolean"
    """Boolean input widget"""

    @staticmethod
    def from_str(value: str) -> "WidgetType":
        """Convert string to widget type"""
        return WidgetType(value)

    def __str__(self) -> str:
        return self.value


class WidgetDefinition(ABC):
    """Base class for widget definitions"""

    type: WidgetType
    """The widget type"""

    def __call__(self) -> Dict[str, Union[str, int, float]]:
        result = {}
        for attr in self.__dir__():
            if not attr.startswith("_") and not callable(self.__getattribute__(attr)):
                result[attr] = self.__getattribute__(attr)
        return result


class TextInput(WidgetDefinition):
    """String input widget"""

    type = WidgetType.STRING

    def __init__(self, max_length=None):
        self.max_length = max_length


class StringInput(WidgetDefinition):
    """String input widget"""

    type = WidgetType.STRING

    def __init__(self, max_length=None):
        self.max_length = max_length


class EnumInput(WidgetDefinition):
    """Enum input widget"""

    type = WidgetType.ENUM

    def __init__(self, options=None):
        if options is None:
            options = []
        self.options = options


class IntInput(WidgetDefinition):
    """Integer input widget"""

    type = WidgetType.INT

    def __init__(self, step=1, default=None, max=None, min=None):
        self.min = min
        self.max = max
        self.step = step
        self.default = default


class FloatInput(WidgetDefinition):
    """Float input widget"""

    type = WidgetType.FLOAT

    def __init__(self, step=1, default=None, max=None, min=None):
        self.min = min
        self.max = max
        self.step = step
        self.default = default


class BooleanInput(WidgetDefinition):
    """Boolean input widget"""

    type = WidgetType.BOOLEAN

    def __init__(self, default=False):
        self.default = default
