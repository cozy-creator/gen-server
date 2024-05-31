import os


folders = {}


base_path = os.path.dirname(os.path.abspath(__file__))

folders["core_nodes"] = os.path.join(base_path, 'extensions', 'core')

def get_folder_paths(folder_name):
    return folders[folder_name]