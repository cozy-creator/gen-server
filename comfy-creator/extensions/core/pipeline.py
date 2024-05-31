import os
import diffusers
from diffusers import StableDiffusionXLPipeline
import torch

def create_pipeline():
    pass

# TO DO 1: Make 'from_single_file' work as you would expect; i.e., load with absolute and relative file paths

# so '../..' will not work
# huggingface_hub.utils._validators.HFValidationError: Repo id must use alphanumeric chars or '-', '_', '.', '--' and '..' are forbidden,
# '-' and '.' cannot start or end the name, max length is 96: 'C:\/git'.
# model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'break_domain_xl.safetensors')
model_path = os.path.join('./', 'models', 'break_domain_xl.safetensors')
pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.float16, repo_type=None)

# TO DO 3: replace this with just a 'load_weights' method. You should specify the file you want to load,
# and optionally a config file. If a config is not specified, a default config is used.
# this 'from_pretrained' versus 'from_single_file' is fucking stupid

# TO DO 2: replace 'save_config' with 'to_config', which returns an object which you serialize and write wherever
# the fuck you want. 'save_config' makes no sense; you don't even get to specify the filename
# TO DO 3: 'config_name' is now pointless

# this outputs a file named 'model_index.json'; placed in the root of your hugging face repo
pipe.save_config('./config/something.json')

# StableDiffusionXLPipeline.extract_init_dict()
# StableDiffusionXLPipeline.config_name
# StableDiffusionXLPipeline.load_config()
# StableDiffusionXLPipeline.from_config()
# StableDiffusionXLPipeline.check_inputs()


