import json
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from .cli_args import args
# from .common.firebase import initialize
# from .settings import settings
from .extension_loader import API_ENDPOINTS, CUSTOM_NODES, WIDGETS, load_components
from gen_server.arch_registry import load_models, ArchDefinition


def main():
    print(API_ENDPOINTS)
    print(CUSTOM_NODES)
    print(WIDGETS)
    print(WIDGETS)
    
    ARCHITECTURES = load_components('comfy_creator.architectures', expected_type=ArchDefinition)
    
    whatever = load_models.from_file("../../models/meinamix.safetensors", 'cpu', ARCHITECTURES)
    
    print(whatever)
    
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
