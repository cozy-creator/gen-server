from dataclasses import is_dataclass, asdict
import collections.abc


# def to_dict(obj):
#     """
#     Convert an object to a dictionary recursively. Checks if the object has a 'to_dict' method and uses it.
#     If not, it checks if the object is a dataclass and uses dataclasses.asdict.
#     Otherwise, it tries to serialize accessible attributes of the object.

#     Args:
#         obj: The object to serialize.

#     Returns:
#         dict: The serialized representation of the object.
#     """
#     if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
#         # Use the custom serialize method if available
#         return obj.to_dict()
#     elif is_dataclass(obj):
#         # Use dataclasses.asdict for dataclass instances
#         return asdict(obj)
#     elif hasattr(obj, '__dict__'):
#         # Fallback to serializing public attributes, recursively handling nested objects
#         return {k: to_dict(v) if isinstance(v, collections.abc.Mapping) or hasattr(v, '__dict__') else v
#                 for k, v in obj.__dict__.items() if not k.startswith('_')}
#     else:
#         # If no suitable serialization method is found, raise an error
#         raise TypeError(f"Object of type {type(obj).__name__} is not convertible to dictionary")


# class ToDictMixin:
#     """
#     Mixin for adding dictionary conversion capabilities to classes
#     """

#     def to_dict(self) -> dict[str, Any]:
#         """
#         Serialize the object into a dictionary
#         :return: serialized object
#         """
#         result = {}
        
#         # Filter attributes:
#         public_attrs = {
#             attr: getattr(self, attr)
#             for attr in dir(self)
#             if not attr.startswith("_") and not callable(getattr(self, attr))
#         }
#         result.update(public_attrs)
        
#         return result