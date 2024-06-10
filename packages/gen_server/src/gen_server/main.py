import os
import sys
import json
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPTokenizer
import time
import inspect
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from .cli_args import args
# from .common.firebase import initialize
# from .settings import settings
from .types import ArchDefinition
from .utils import load_models
from .utils.extension_loader import load_extensions
from .globals import API_ENDPOINTS, ARCHITECTURES,  CUSTOM_NODES, WIDGETS

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../models/meinamix.safetensors"))
output_folder = os.path.join(os.path.dirname(__file__), "../../../../output")


def main():
    # we load the extensions inside a function to avoid circular dependencies
    
    # Api-endpoints will extend the aiohttp rest server somehow
    # Architectures will be classes that can be used to detect models and instantiate them
    # custom nodes will define new nodes to be instantiated by the graph-editor
    # widgets will somehow define react files to be somehow be imported by the client
    
    global API_ENDPOINTS
    API_ENDPOINTS = load_extensions('comfy_creator.api')
    
    global ARCHITECTURES
    ARCHITECTURES = load_extensions('comfy_creator.architectures', expected_type=ArchDefinition)
    
    global CUSTOM_NODES
    CUSTOM_NODES = load_extensions('comfy_creator.custom_nodes')
    
    global WIDGETS
    WIDGETS = load_extensions('comfy_creator.widgets')

    # print(API_ENDPOINTS)
    print (ARCHITECTURES)
    # print(CUSTOM_NODES)
    # print(WIDGETS)

    # models = load_models.from_file(file_path, 'cpu', ARCHITECTURES)
    
    # === Siimulating the executor code ==
    LoadCheckpoint = CUSTOM_NODES["core_extension_1.load_checkpoint"]
    load_checkpoint = LoadCheckpoint()
    
    # Return this to the UI
    architectures = load_checkpoint.determine_output(file_path)
    # print(architectures)
    
    # execute the first node
    models = load_checkpoint(file_path)
    print("Number of items loaded:", len(models))
    for model_key in models.keys():
        print(f"Model key: {model_key}")
    
    models = load_models.from_file(file_path)
    print("Number of items loaded:", len(models))
    for model_key in models.keys():
        print(f"Model key: {model_key}")
    
    # load node 2
    CreatePipe = CUSTOM_NODES["core_extension_1.create_pipe"]
    create_pipe = CreatePipe()
    
    # ???
    # pipe_type = create_pipe.determine_output()
    # print(pipe_type)
    
    signature = inspect.signature(create_pipe.__call__)
    # print(signature)
    
    # Detailed parameter analysis
    for name, param in signature.parameters.items():
        print(f"Parameter Name: {name}")
        print(f"  Kind: {param.kind}")
        print(f"  Default: {param.default if param.default is not inspect.Parameter.empty else 'No default'}")
        print(f"  Annotation: {param.annotation if param.annotation is not inspect.Parameter.empty else 'No annotation'}")
    
    # how do we know this? Edges?
    vae = models["core_extension_1.sd1_vae"]
    text_encoder = models["core_extension_1.sd1_text_encoder"]
    unet = models["core_extension_1.sd1_unet"]
    
    # run node 2
    pipe = create_pipe(vae=vae, text_encoder=text_encoder, unet=unet)
    
    # node 3
    run_pipe = CUSTOM_NODES["core_extension_1.run_pipe"]()
    
    # ???
    # output_type = run_pipe.determine_output()
    # print(output_type)
    
    start = time.time()
    
    # execute the 3rd node
    prompt = "beautiful anime woman, detailed, masterpiece, dark skin"
    negative_prompt = "poor quality, worst quality, text, watermark, blurry"
    images = run_pipe(pipe, prompt=prompt, negative_prompt=negative_prompt)
    

    # diffusion_pytorch_model.fp16.safetensors
    # playground-v2.5-1024px-aesthetic.fp16.safetensors
    # diffusion_pytorch_model.safetensors
    # darkSushi25D25D_v40.safetensors
    
    for idx, img in enumerate(images):
        img.save(os.path.join(output_folder, f"generated_image_{idx}.png"))
    
    print(f"Image generated in {time.time() - start} seconds")
    
    # if args.run_web_server:
    #     from request_handlers.web_server import start_server

    # if args.run_web_server:
    #     from request_handlers.web_server import start_server
    #
    #     start_server(args.host, args.web_server_port)

    # if args.run_grpc:
    #     from request_handlers.grpc_server import start_server

    #     start_server(args.host, args.grpc_port)


if __name__ == "__main__":
    # initialize(json.loads(settings.firebase.service_account))
    main()
