import os
import json
import time
import inspect
from dotenv import load_dotenv
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from .cli_args import args
# from .common.firebase import initialize
# from .settings import settings
from .base_types import Architecture, CustomNode
from .utils.extension_loader import load_extensions
from .globals import (
    API_ENDPOINTS,
    ARCHITECTURES,
    CUSTOM_NODES,
    WIDGETS,
    PRETRAINED_MODELS,
    initialize_config,
)
import argparse
import ast
from aiohttp import web
from api import app

file_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "../../../../models/sd3_medium_incl_clips_t5xxlfp8.safetensors"
    )
)


def main():
    # Parse command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="Environment file path", default=None)
    parser.add_argument(
        "--config", help="Configuration dictionary (JSON format)", default=None
    )
    args = parser.parse_args()

    initialize_config(env_path=args.env, config_path=args.config)

    # We load the extensions inside a function to avoid circular dependencies

    # Api-endpoints will extend the aiohttp rest server somehow
    # Architectures will be classes that can be used to detect models and instantiate them
    # custom nodes will define new nodes to be instantiated by the graph-editor
    # widgets will somehow define react files to be somehow be imported by the client
    
    global API_ENDPOINTS
    API_ENDPOINTS.update(load_extensions("comfy_creator.api"))
    
    # compile architecture registry
    global ARCHITECTURES
    ARCHITECTURES.update(
        load_extensions("comfy_creator.architectures", expected_type=Architecture)
    )

    global CUSTOM_NODES
    CUSTOM_NODES.update(
        load_extensions("comfy_creator.custom_nodes", expected_type=CustomNode)
    )

    global WIDGETS
    WIDGETS.update(load_extensions("comfy_creator.widgets"))
    
    # compile model registry
    global PRETRAINED_MODELS
    # to do

    # print(API_ENDPOINTS)
    # print (ARCHITECTURES)
    # print(CUSTOM_NODES)
    # print(WIDGETS)

    # models = load_models.from_file(file_path, 'cpu', ARCHITECTURES)

    # === Simulating the executor code ===
    LoadCheckpoint = CUSTOM_NODES["core_extension_1.load_checkpoint"]

    # Return this to the UI
    architectures = LoadCheckpoint.update_interface(
        {"inputs": {"file_path": file_path}}
    )
    # print(architectures)

    # figure out what outputs we need from this node
    output_keys = {}
    load_checkpoint = LoadCheckpoint()

    # execute the first node
    models = load_checkpoint(file_path, output_keys=output_keys)

    # print(models)

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
    # for name, param in signature.parameters.items():
    #     print(f"Parameter Name: {name}")
    #     print(f"  Kind: {param.kind}")
    #     print(f"  Default: {param.default if param.default is not inspect.Parameter.empty else 'No default'}")
    #     print(f"  Annotation: {param.annotation if param.annotation is not inspect.Parameter.empty else 'No annotation'}")

    # how do we know this? Edges?
    # SD3
    vae = models["core_extension_1.sd1_vae"].model
    unet = models["core_extension_1.sd3_unet"].model
    text_encoder_1 = models["core_extension_1.sd3_text_encoder_1"].model
    text_encoder_2 = models["core_extension_1.sd3_text_encoder_2"].model
    text_encoder_3 = models["core_extension_1.sd3_text_encoder_3"].model

    # # run node 2
    # pipe = create_pipe(
    #     vae=vae, 
    #     text_encoder=text_encoder_1,
    #     text_encoder_2=text_encoder_2,
    #     text_encoder_3=text_encoder_3,
    #     unet=unet
    # )

    # SD1.5
    # vae = models["core_extension_1.sd1_vae"].model
    # unet = models["core_extension_1.sd1_unet"].model
    # text_encoder_1 = models["core_extension_1.sd1_text_encoder"].model

    pipe = create_pipe(
        vae=vae, 
        text_encoder=text_encoder_1,
        text_encoder_2=text_encoder_2,
        text_encoder_3=text_encoder_3,
        unet=unet
    )
    # pipe.to('cuda')

    # node 3
    run_pipe = CUSTOM_NODES["core_extension_1.run_pipe"]()

    # ???
    # output_type = run_pipe.determine_output()
    # print(output_type)

    start = time.time()

    # execute the 3rd node
    prompt = "Beautiful anime woman with dark-skin"
    negative_prompt = "poor quality, worst quality, watermark, blurry"
    images = run_pipe(pipe, prompt=prompt, negative_prompt=negative_prompt, width=1024, height=1024)

    # Save Images
    # images[0].save("output.png")

    # diffusion_pytorch_model.fp16.safetensors
    # playground-v2.5-1024px-aesthetic.fp16.safetensors
    # diffusion_pytorch_model.safetensors
    # darkSushi25D25D_v40.safetensors
    # sd3_medium_incl_clips_t5xxlfp8.safetensors

    # Save the images
    SaveNode = CUSTOM_NODES["image_utils.save_file"]
    save_node = SaveNode()
    save_node(images=images, temp=False)

    # for idx, img in enumerate(images):
    #     img.save(os.path.join(output_folder, f"generated_image_{idx}.png"))

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
    
    # Run our REST server
    web.run_app(app, port=8080)
