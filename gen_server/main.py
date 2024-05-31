import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cli_args import args
from gen_server.common.firebase import initialize
from gen_server.settings import settings

if __name__ == "__main__":
    initialize(json.loads(settings.firebase.service_account))

    if args.run_web_server:
        from request_handlers.web_server import start_server

    # if args.run_web_server:
    #     from request_handlers.web_server import start_server
    #
    #     start_server(args.host, args.web_server_port)

    if args.run_grpc:
        from request_handlers.grpc_server import start_server

        start_server(args.host, args.grpc_port)
