# from .paths import get_folder_path
import sys
import logging
from typing import TypeVar, Optional

from ..base_types.common import Validator
from .utils import to_snake_case
from ..base_types import CustomNode

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


def load_custom_node_specs(
    custom_nodes: dict[str, type[CustomNode]],
) -> dict[str, dict]:
    custom_node_specs = {}

    for node_name, node_class in custom_nodes.items():
        try:
            if hasattr(node_class, "get_spec") and callable(node_class.get_spec):
                spec = node_class.get_spec()
                custom_node_specs[node_name] = spec
        except Exception as e:
            logging.error(f"Failed to get spec for custom node {node_name}: {str(e)}")

    return custom_node_specs


T = TypeVar("T")


def load_extensions(
    entry_point_group: str, validator: Optional[Validator] = None
) -> dict[str, T]:
    discovered_plugins = entry_points(group=entry_point_group)
    plugins: dict[str, T] = {}

    for entry_point in discovered_plugins:
        # Scope the plugin's name using the distribution name; ex. 'cozy_creator.sdxl' rather than just 'sdxl'
        try:
            assert (
                entry_point.dist is not None
            ), "The distribution object for the entry point is None."
            package_name = entry_point.dist.metadata["Name"].replace("-", "_")
            scoped_name = f"{package_name}.{entry_point.name}"
        except AssertionError as e:
            logging.error(
                f"Error in processing entry point {entry_point.name}: {str(e)}"
            )
            continue  # Skip this entry point
        try:

            def _load_plugin_inner(plugin_name: str, plugin_item: T):
                # Optionally validate the plugin, if a validator is provided
                if validator is not None and not validator(plugin_item):
                    logging.error(f'Failed to validate plugin "{plugin_name}" type.')
                    raise ValueError(f"Invalid plugin type for {plugin_name}")

                # print(f"Loading plugin {plugin_name}")
                plugins[plugin_name] = plugin_item

            plugin = entry_point.load()

            if isinstance(plugin, list) and all(
                isinstance(item, type) for item in plugin
            ):
                for item in plugin:
                    scoped_item_name = f"{scoped_name}.{to_snake_case( item.__name__)}"
                    _load_plugin_inner(scoped_item_name, item)
            else:
                _load_plugin_inner(scoped_name, plugin)
        except Exception as error:
            logging.error(f"Failed to load plugin {scoped_name}: {str(error)}")

    return plugins
