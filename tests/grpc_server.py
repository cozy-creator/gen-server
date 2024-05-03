import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../gen_server')))

from gen_server.grpc_server import ComfyServicer
from protobuf.node_defs_v1_pb2 import NodeDefRequest

def test_local_method_call():
    servicer = ComfyServicer()
    request = NodeDefRequest()
    context = None
    response = servicer.GetNodeDefinitions(request, context)
    print("Direct method call response:", response)

if __name__ == '__main__':
    test_local_method_call()

