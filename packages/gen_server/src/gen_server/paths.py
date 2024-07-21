import os
import shutil

from gen_server.templates.env import ENV_TEMPLATE

base_path = os.path.dirname(os.path.abspath(__file__))

folders = {
    "core_nodes": os.path.join(base_path, "extensions", "core"),
    "extensions": os.path.join(base_path, "extensions"),
    "output": os.path.join(base_path, "output"),
    "temp": os.path.join(base_path, "temp"),
    "input": os.path.join(base_path, "input"),
    "models": os.path.join(base_path, "models"),
    "custom_architecture": os.path.join(base_path, "custom_architecture"),
    "vae": os.path.join(base_path, "extensions/core2/VAE"),
    "unet": os.path.join(base_path, "extensions/core2/unet"),
    "text_encoder": os.path.join(base_path, "extensions/core2/text_encoder"),
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


def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0):
    def map_filename(filename):
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[: prefix_len + 1]
        try:
            digits = int(filename[prefix_len + 1 :].split("_")[0])
        except:
            digits = 0
        return (digits, prefix)

    def compute_vars(input, image_width, image_height):
        input = input.replace("%width%", str(image_width))
        input = input.replace("%height%", str(image_height))
        return input

    filename_prefix = compute_vars(filename_prefix, image_width, image_height)

    subfolder = os.path.dirname(os.path.normpath(filename_prefix))
    filename = os.path.basename(os.path.normpath(filename_prefix))

    full_output_folder = os.path.join(output_dir, subfolder)

    if (
        os.path.commonpath((output_dir, os.path.abspath(full_output_folder)))
        != output_dir
    ):
        err = (
            "**** ERROR: Saving image outside the output folder is not allowed."
            + "\n full_output_folder: "
            + os.path.abspath(full_output_folder)
            + "\n         output_dir: "
            + output_dir
            + "\n         commonpath: "
            + os.path.commonpath((output_dir, os.path.abspath(full_output_folder)))
        )
        print(err)
        raise Exception(err)

    try:
        counter = (
            max(
                filter(
                    lambda a: a[1][:-1] == filename and a[1][-1] == "_",
                    map(map_filename, os.listdir(full_output_folder)),
                )
            )[0]
            + 1
        )
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
    return full_output_folder, filename, counter, subfolder, filename_prefix


def get_model_path(folder_name):
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


def annotated_filepath(name):
    if name.endswith("[output]"):
        base_dir = get_folder_path("output")
        name = name[:-9]
    elif name.endswith("[input]"):
        base_dir = get_folder_path("input")
        name = name[:-8]
    elif name.endswith("[temp]"):
        base_dir = get_folder_path("temp")
        name = name[:-7]
    else:
        return name, None

    return name, base_dir


def get_annotated_filepath(name, default_dir=None):
    name, base_dir = annotated_filepath(name)

    if base_dir is None:
        if default_dir is not None:
            base_dir = default_dir
        else:
            base_dir = get_folder_path("input")  # fallback path

    return os.path.join(base_dir, name)


def exists_annotated_filepath(name):
    name, base_dir = annotated_filepath(name)

    if base_dir is None:
        base_dir = get_folder_path("input")  # fallback path

    filepath = os.path.join(base_dir, name)
    return os.path.exists(filepath)


def check_model_in_path(model_id, model_path):
    if model_id.endswith(".safetensors") or model_id.endswith(".ckpt"):
        if os.path.exists(os.path.join(model_path, model_id)):
            return os.path.join(model_path, model_id)
        else:
            raise ValueError(f"File not found: {model_id}")
    else:
        return None  # Treat the model_id as a repository ID


def ensure_workspace_path(path: str):
    subdirs = ["models", ["assets", "temp"]]

    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    for subdir in subdirs:
        if isinstance(subdir, list):
            subdir_path = os.path.join(path, *subdir)
        else:
            subdir_path = os.path.join(path, subdir)

        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path, exist_ok=True)

    # We also ensure the .env file exists in the workspace
    ensure_env_file(path)


def ensure_env_file(workspace_path: str):
    try:
        env_path = os.path.expanduser(os.path.join(workspace_path, ".env"))
        if not os.path.exists(env_path):
            with open(env_path, "w") as f:
                f.write(ENV_TEMPLATE)

    except Exception as e:
        print(f"Error while creating initializing env file: {str(e)}")


def clean_temp_files(workspace_path: str):
    temp_path = os.path.join(workspace_path, "temp")
    if not os.path.exists(temp_path):
        return

    def _clean_files(files: list[str]):
        for file in files:
            file_path = os.path.join(temp_path, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error while deleting {file_path}: {str(e)}")

    def _clean_dirs(dirs: list[str]):
        for dir in dirs:
            dir_path = os.path.join(temp_path, dir)
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                print(f"Error while deleting {dir_path}: {str(e)}")

    for _, dirs, files in os.walk(temp_path, topdown=False):
        _clean_dirs(dirs)
        _clean_files(files)
