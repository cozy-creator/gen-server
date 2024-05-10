import os
import sys
from concurrent import futures

from gen_server.settings import settings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../protobuf')))

import grpc
from google.protobuf import empty_pb2
from uuid import uuid4
from protobuf import ComfyGRPCServiceServicer, add_ComfyGRPCServiceServicer_to_server
from protobuf import ComfyRequest, JobSnapshot, JobStatus, JobId, JobIds, UserId, UserHistory, NodeDefRequest, NodeDefs, \
    NodeDefinition, ModelCatalogRequest, ModelCatalog, Models, LocalFiles, LocalFile, Workflow
from gen_server.cqueue.pulsar import add_topic_message


class ComfyServicer(ComfyGRPCServiceServicer):
    def Run(self, request: ComfyRequest, context: grpc.ServicerContext) -> JobSnapshot:
        properties = {
            "job_id": uuid4(),
            "user_uid": context.invocation_metadata().get("user_id"),
            "estimated_duration": 1000,
            "publish_outputs_to_topic": True
        }

        try:
            add_request_to_queue(request, properties)
        except Exception as e:
            print(f"Error adding request to queue: {e}")
            context.abort(grpc.StatusCode.INTERNAL, "Error adding request to queue")

        return JobSnapshot(
            job_id=properties['job_id'].__str__(),
            request_id=request.request_id,
            status=JobStatus.QUEUED
        )

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
        node = NodeDefinition(display_name="whatever", description="whatever", category="none", ux_widgets=[],
                              inputs=[], outputs=[])
        node_defs = NodeDefs(defs={"test": node})

        return node_defs

    def GetModelCatalog(self, request: ModelCatalogRequest, context: grpc.ServicerContext) -> ModelCatalog:
        return ModelCatalog(models={"base": Models(info=[Models.Info(blake3_hash="hash1")])})

    def SyncLocalFiles(self, request: empty_pb2.Empty, context: grpc.ServicerContext) -> LocalFiles:
        yield LocalFiles(added=[LocalFile(name="file1.txt")])


def add_request_to_queue(request: ComfyRequest, properties: dict):
    topic = get_affinity(request.workflow)
    add_topic_message(topic, message=request.SerializeToString(), properties=properties,
                      namespace=settings.pulsar.job_queue_namespace)


def get_affinity(_request: Workflow):
    return "sdxl-1"


def _unary_unary_rpc_terminator(code, details):
    def terminate(ignored_request, context):
        context.abort(code, details)

    return grpc.unary_unary_rpc_method_handler(terminate)


class AuthenticationInterceptor(grpc.ServerInterceptor):
    def __init__(self):
        self._terminator = _unary_unary_rpc_terminator(grpc.StatusCode.UNAUTHENTICATED, "Invalid auth details")

    def intercept_service(self, continuation, handler_call_details):
        for key, value in handler_call_details.invocation_metadata:
            if key.lower() == 'authorization':
                parts = value.split(maxsplit=1)
                if parts[0] != 'Bearer':
                    raise ValueError('Invalid bearer token')

                # TODO: Authenticate token here
                user_id = verify_token(parts[1])
                if not user_id:
                    return self._terminator

                handler_call_details.invocation_metadata = handler_call_details.invocation_metadata + (
                    'user_id', user_id)

                return continuation(handler_call_details)
        else:
            return self._terminator


def verify_token(token: str):
    # TODO: Implement token verification
    return "user_123"


def start_server(_host: str, port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), interceptors=(AuthenticationInterceptor()))
    add_ComfyGRPCServiceServicer_to_server(ComfyServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print("Server started. Listening on port 50051.")
    server.wait_for_termination()
