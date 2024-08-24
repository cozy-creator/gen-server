from typing import List


class WidgetDefinition:
    """Base class for widget definitions"""

    def __init__(self, value=None, default=None):
        if value is None and default is not None:
            value = default

        self.default = default
        self.value = value

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

    def __init__(self, default=None, value=None, max_length=None):
        super().__init__(value, default)
        self.max_length = max_length


class StringInput(WidgetDefinition):
    """String input widget"""

    def __init__(self, default=None, value=None, max_length=None):
        super().__init__(value, default)
        self.max_length = max_length


class EnumInput(WidgetDefinition):
    """Enum input widget"""

    def __init__(self, default=None, value=None, options: List[str] = []):
        if default is not None and default not in options:
            raise ValueError(f"Invalid default value: {default}")
        if value is not None and value not in options:
            raise ValueError(f"Invalid value: {value}")
        if not all(isinstance(option, str) for option in options):
            raise TypeError("All options must be strings.")

        super().__init__(value, default)

        self.options = options


class IntInput(WidgetDefinition):
    """Integer input widget"""

    def __init__(self, default=None, value=None, step=1, max=None, min=None):
        super().__init__(value, default)
        self.min = min
        self.max = max
        self.step = step


class FloatInput(WidgetDefinition):
    """Float input widget"""

    def __init__(self, default=None, value=None, step=1, max=None, min=None):
        super().__init__(value, default)
        self.min = min
        self.max = max
        self.step = step


class BooleanInput(WidgetDefinition):
    """Boolean input widget"""

    def __init__(
        self,
        default=None,
        value=None,
    ):
        super().__init__(value, default)
