import os

base_path = os.path.dirname(os.path.abspath(__file__))

folders = {
    "core_nodes": os.path.join(base_path, 'extensions', 'core'),
    "extensions": os.path.join(base_path, 'extensions'),
    "output": os.path.join(base_path, 'output'),
    "temp": os.path.join(base_path, 'temp'),
}

def get_folder_path(folder_name):
    """
    Returns the path for the given folder name.
    Creates the folder if it doesn't exist.
    """
    folder_path = folders.get(folder_name)
    if folder_path is None:
        raise ValueError(f"Invalid folder name: {folder_name}")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return folder_path
