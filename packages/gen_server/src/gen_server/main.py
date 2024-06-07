import json
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from .cli_args import args
# from .common.firebase import initialize
# from .settings import settings
from .extension_loader import API_ENDPOINTS, CUSTOM_NODES, WIDGETS
from gen_server.arch_registry import load_models


def main():
    print(API_ENDPOINTS)
    print(CUSTOM_NODES)
    print(WIDGETS)
    
    whatever = load_models.from_file("../../models/meinamix.safetensors")
    
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
