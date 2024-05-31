import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--host", type=str, default="127.0.0.1",
                    help="Specify the IP address which the server listens on [default: 127.0.0.1].")
parser.add_argument("---web-server-port", type=int, default=5055,
                    help="Specify port which the web server listens on [default: 5055].")
parser.add_argument("--grpc-port", type=int, default=50051,
                    help="Specify port which the gRPC server listens on [default: 50051].")
parser.add_argument("--run-web-server", action="store_true", default=True,
                    help="Specify whether to run the web server.")
parser.add_argument("--run-grpc", action="store_true", default=True, help="Specify whether to run the gRPC server.")

args = parser.parse_args()
