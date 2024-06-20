from .load_models import load_state_dict_from_file


def load_from_file(path: str, device):
    state_dict = load_state_dict_from_file(path, device)
