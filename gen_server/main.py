from cli_args import args
from core_library.firebase import initialize

if __name__ == "__main__":
    # if args.run_web_server:
    #     from request_handlers.web_server import start_server
    #
    #     start_server(args.host, args.web_server_port)

    initialize("/Users/byte/Documents/Capsules/Gen-Server/service_account.json")
    if args.run_grpc:
        from request_handlers.grpc_server import start_server

        start_server(args.host, args.grpc_port)
