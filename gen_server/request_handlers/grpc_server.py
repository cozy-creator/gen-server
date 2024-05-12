import os
import sys
from concurrent import futures
from contextvars import ContextVar
from pprint import pprint
from typing import Optional

from core_library.firebase import verify_token
from gen_server.settings import settings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../protobuf')))

import grpc
from google.protobuf import empty_pb2
from uuid import uuid4
from protobuf import ComfyGRPCServiceServicer, add_ComfyGRPCServiceServicer_to_server
from protobuf import ComfyRequest, JobSnapshot, JobStatus, JobId, JobIds, UserId, UserHistory, NodeDefRequest, NodeDefs, \
    NodeDefinition, ModelCatalogRequest, ModelCatalog, Models, LocalFiles, LocalFile, Workflow
from gen_server.cqueue.pulsar import add_topic_message, create_subscription

user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


class ComfyServicer(ComfyGRPCServiceServicer):
    def Run(self, request: ComfyRequest, context: grpc.ServicerContext) -> JobSnapshot:
        global job_id
        if user_id_var.get() is None:
            return context.abort(grpc.StatusCode.UNAUTHENTICATED, "Authentication Required")

        try:
            job_id = handle_job_request(request, user_id_var.get())
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, "Failed to add job to queue")

        return JobSnapshot(
            job_id=job_id,
            status=JobStatus.QUEUED,
            request_id=request.request_id
        )

    def RunSync(self, request: ComfyRequest, context: grpc.ServicerContext) -> JobSnapshot:
        global job_id
        if user_id_var.get() is None:
            return context.abort(grpc.StatusCode.UNAUTHENTICATED, "Authentication Required")

        try:
            job_id = handle_job_request(request, user_id_var.get())
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, "Failed to add job to queue")

        success = wait_for_job_completion(job_id)
        if not success:
            context.abort(grpc.StatusCode.INTERNAL, "Failed to process job")

        job_snapshot = JobSnapshot(
            job_id=job_id,
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
    namespace = settings.pulsar.job_queue_namespace

    message_id = add_topic_message(
        topic,
        namespace=namespace,
        properties=properties,
        message=request.SerializeToString()
    )

    if not message_id:
        raise Exception("Failed to add job to queue")

    return message_id


def handle_job_request(request: ComfyRequest, user_id: str):
    properties = {
        "job_id": str(uuid4()),
        "user_uid": user_id_var.get(),
        "estimated_duration": str(1000),
        "publish_outputs_to_topic": str(True)
    }

    try:
        message_id = add_request_to_queue(request, properties)
        print(f"New message id: {message_id}")
    except Exception as e:
        print(f"Error adding request to queue: {e}")
        raise Exception("Failed to add job to queue")

    return properties['job_id']


def wait_for_job_completion(job_id: str):
    def listener(message, consumer):
        print(f"Received message: {message}")
        consumer.acknowledge(message)

    topic = f"non-persistent://{settings.pulsar.tenant}/job-output/{job_id}"
    create_subscription(topic, listener)

    return True


def get_affinity(_workflow: Workflow):
    return "sdxl-1"


def _unary_unary_rpc_terminator(code, details):
    def terminate(ignored_request, context):
        context.abort(code, details)

    return grpc.unary_unary_rpc_method_handler(terminate)


class AuthenticationInterceptor(grpc.ServerInterceptor):
    def __init__(self):
        self._header = 'authorization'
        self._terminator = _unary_unary_rpc_terminator(grpc.StatusCode.UNAUTHENTICATED, "Invalid auth details")

    def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata)
        if self._header in metadata:
            value = metadata.get(self._header)
            if value is None:
                return self._terminator

            bearer_parts = value.split(maxsplit=1)
            if bearer_parts[0] != 'Bearer':
                return self._terminator

            user = verify_token(bearer_parts[1])
            if not user:
                return self._terminator

            if user_id_var.get() is not None:
                print("Corrupted context var")

            user_id_var.set(user['uid'])
            return continuation(handler_call_details)
        else:
            return self._terminator


def start_server(_host: str, port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), interceptors=(AuthenticationInterceptor(),))
    add_ComfyGRPCServiceServicer_to_server(ComfyServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print("Server started. Listening on port 50051.")
    server.wait_for_termination()
