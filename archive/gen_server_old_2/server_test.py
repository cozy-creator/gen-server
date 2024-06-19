from pprint import pprint

import grpc

from gen_server.proto_defs import (
    NodeDefRequest,
    ComfyGRPCServiceStub,
)


def create_channel():
    yield grpc.insecure_channel("localhost:50051")


# def run_job():
#     with grpc.insecure_channel("localhost:50051") as channel:
#         comfy_stub = ComfyGRPCServiceStub(channel)
#         request = ComfyRequest(request_id=str(uuid.uuid4()), serialized_graph=None)
#
#         try:
#             response = comfy_stub.Run.with_call(request, metadata=[auth_metadata])
#             print("----- Run job -----")
#             return response[0]
#         except grpc.RpcError as rpc_error:
#             print("Received error: %s", rpc_error)
#             return rpc_error


def get_node_defs():
    with grpc.insecure_channel("localhost:50051") as channel:
        comfy_stub = ComfyGRPCServiceStub(channel)
        request = NodeDefRequest(extension_ids=[])

        try:
            response = comfy_stub.GetNodeDefinitions(request)
            print("----- Received node definitions -----")
            pprint(response)
            return response
        except grpc.RpcError as rpc_error:
            print("Received error: %s", rpc_error)
            return rpc_error


if __name__ == "__main__":
    get_node_defs()
