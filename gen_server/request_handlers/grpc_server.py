import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../protobuf')))

import grpc
from google.protobuf import empty_pb2
from protobuf import JobSnapshot, JobStatus, UserHistory, NodeDefs, ModelCatalog, LocalFiles, LocalFile, Models
from protobuf import ComfyGRPCServiceServicer, add_ComfyGRPCServiceServicer_to_server
from protobuf import ComfyRequest, JobSnapshot, JobStatus, JobId, JobIds, UserId, UserHistory, NodeDefRequest, NodeDefs, NodeDefinition, ModelCatalogRequest, ModelCatalog, Models, LocalFiles, LocalFile
from concurrent import futures


class ComfyServicer(ComfyGRPCServiceServicer):
    def Run(self, request: ComfyRequest, context: grpc.ServicerContext) -> JobSnapshot:
        job_snapshot = JobSnapshot(
            job_id="job_123",
            request_id=request.request_id,
            status=JobStatus.QUEUED
        )
        return job_snapshot

    def RunSync(self, request: ComfyRequest, context: grpc.ServicerContext) -> JobSnapshot:
        job_snapshot = JobSnapshot(
            job_id="job_456",
            request_id=request.request_id,
            status=JobStatus.COMPLETED
        )
        return job_snapshot

    def Stream(self, request: ComfyRequest, context: grpc.ServicerContext) -> JobSnapshot:
        yield JobSnapshot(job_id="job_789", status=JobStatus.EXECUTING)

    def CancelJob(self, request: JobId, context: grpc.ServicerContext) -> empty_pb2.Empty:
        return empty_pb2.Empty()

    def GetJob(self, request: JobIds, context: grpc.ServicerContext) -> JobSnapshot:
        for job_id in request.job_ids:
            yield JobSnapshot(job_id=job_id, status=JobStatus.COMPLETED)

    def GetUserHistory(self, request: UserId, context: grpc.ServicerContext) -> UserHistory:
        return UserHistory(jobs=[UserHistory.Job(id="job_abc"), UserHistory.Job(id="job_def")])

    def GetNodeDefinitions(self, request: NodeDefRequest, context: grpc.ServicerContext) -> NodeDefs:
        node = NodeDefinition(display_name="whatever", description="whatever", category="none", ux_widgets=[], inputs=[], outputs=[])
        node_defs = NodeDefs(defs={ "test": node })
        
        return node_defs

    def GetModelCatalog(self, request: ModelCatalogRequest, context: grpc.ServicerContext) -> ModelCatalog:
        return ModelCatalog(models={"base": Models(info=[Models.Info(blake3_hash="hash1")])})

    def SyncLocalFiles(self, request: empty_pb2.Empty, context: grpc.ServicerContext) -> LocalFiles:
        yield LocalFiles(added=[LocalFile(name="file1.txt")])

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_ComfyGRPCServiceServicer_to_server(ComfyServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started. Listening on port 50051.")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()

