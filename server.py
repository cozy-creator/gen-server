

# import os
# import sys
# AUTOGEN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'proto/autogen_python')
# if AUTOGEN_PATH not in sys.path:
#     sys.path.append(AUTOGEN_PATH)

# import grpc
# from google.protobuf import empty_pb2
# from concurrent import futures
# from typing import TypedDict, Optional, Dict

from autogen_python.comfy_request.v1_pb2 import JobSnapshot, JobStatus
from autogen_python.comfy_request.v1_pb2_grpc import ComfyServicer
from autogen_python.node_defs.v1_pb2 import NodeDefs

class ComfyServicer(comfy_request_pb2_grpc.ComfyServicer):
    def Run(self, request, context):
        # Implement the Run method
        job_snapshot = comfy_request_pb2.JobSnapshot(
            job_id="job_123",
            request_id=request.request_id,
            status=comfy_request_pb2.JobStatus.QUEUED
        )
        return job_snapshot

    def RunSync(self, request, context):
        # Implement the RunSync method
        job_snapshot = comfy_request_pb2.JobSnapshot(
            job_id="job_456",
            request_id=request.request_id,
            status=comfy_request_pb2.JobStatus.COMPLETED
        )
        return job_snapshot

    def Stream(self, request, context):
        # Implement the Stream method
        output_diff = comfy_request_pb2.OutputDiff(diff=b"output_diff_data")
        yield output_diff

    def GetJob(self, request, context):
        # Implement the GetJob method
        job_snapshot = comfy_request_pb2.JobSnapshot(
            job_id=request.job_id,
            status=comfy_request_pb2.JobStatus.EXECUTING
        )
        return job_snapshot

    def CancelJob(self, request, context):
        # Implement the CancelJob method
        return empty_pb2.Empty()

    def GetNodeDefinitions(self, request, context):
        # Implement the GetNodeDefinitions method
        node_defs = node_defs_pb2.NodeDefs(node_defs=["node1", "node2"])
        return node_defs

    def GetModelCatalog(self, request, context):
        # Implement the GetModelCatalog method
        model_catalog = comfy_request_pb2.ModelCatalog(
            models={
                "base_family1": comfy_request_pb2.Models(
                    info=[
                        comfy_request_pb2.Models.Info(
                            blake3_hash="hash1",
                            display_name="Model 1"
                        )
                    ]
                )
            }
        )
        return model_catalog

    def SyncLocalFiles(self, request, context):
        # Implement the SyncLocalFiles method
        local_files = comfy_request_pb2.LocalFiles(
            added=[comfy_request_pb2.LocalFile(name="file1.txt", path="/path/to/file1.txt")],
            updated=[comfy_request_pb2.LocalFile(name="file2.txt", path="/path/to/file2.txt")],
            removed=[comfy_request_pb2.LocalFile(name="file3.txt", path="/path/to/file3.txt")]
        )
        yield local_files

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    comfy_request_pb2_grpc.add_ComfyServicer_to_server(ComfyServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started. Listening on port 50051.")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
