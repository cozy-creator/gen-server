import os
from typing import TypedDict, Set, Any

class ArchitectureDef(TypedDict):
    display_name: str
    state_dict_set: Set[str]

def load_state_dict_keys(path: str) -> set[str]:
    with open(path, 'r') as file:
        state_dict_keys = file.readlines()
    state_dict_keys = {key.strip() for key in state_dict_keys}
    return state_dict_keys

path = os.path.join(os.path.dirname(__file__), "state_dict_keys.md")

architecture_def: ArchitectureDef = {
    'display_name': 'SDXL',
    'state_dict_set': load_state_dict_keys(path)
}
