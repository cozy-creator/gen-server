import os
import sys
from concurrent import futures
from contextvars import ContextVar
from pprint import pprint
from typing import Optional

import pulsar

from gen_server.common.readers import JobReader
from gen_server.common.firebase import verify_token, Collection
from gen_server.settings import settings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../protobuf')))

import grpc
from google.protobuf import empty_pb2
from uuid import uuid4
from protobuf import ComfyGRPCServiceServicer, add_ComfyGRPCServiceServicer_to_server
from protobuf import ComfyRequest, JobSnapshot, JobStatus, JobId, JobIds, UserId, UserHistory, NodeDefRequest, NodeDefs, \
    NodeDefinition, ModelCatalogRequest, ModelCatalog, Models, LocalFiles, LocalFile, Workflow
from gen_server.job_queue.pulsar import add_topic_message, Pulsar, make_topic

context_user_uid: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
queue = Pulsar(settings.pulsar)


class ComfyServicer(ComfyGRPCServiceServicer):
    def Run(self, request: ComfyRequest, context: grpc.ServicerContext) -> JobSnapshot:
        _ensure_context_user(context)
        job_id = _send_job_request(request, context)

        return JobSnapshot(
            job_id=job_id,
            status=JobStatus.QUEUED,
            request_id=request.request_id
        )

    def RunSync(self, request: ComfyRequest, context: grpc.ServicerContext) -> JobSnapshot:
        _ensure_context_user(context)
        job_id = _send_job_request(request, context)

        try:
            snapshot = _wait_for_job_snapshot(job_id)
            if not snapshot:
                return context.abort(grpc.StatusCode.INTERNAL, "Failed to process job")
            return snapshot
        except Exception as e:
            return context.abort(grpc.StatusCode.INTERNAL, "Failed to retrieve job snapshot")

    def Stream(self, request: ComfyRequest, context: grpc.ServicerContext) -> JobSnapshot:
        _ensure_context_user(context)
        job_id = _send_job_request(request, context)

        try:
            for snapshots in stream_jobs(job_id):
                yield snapshots
        except Exception as e:
            return context.abort(grpc.StatusCode.INTERNAL, "Failed to stream job snapshots")

    def CancelJob(self, request: JobId, context: grpc.ServicerContext) -> empty_pb2.Empty:
        _ensure_context_user(context)
        _request_job_cancellation(request)
        return empty_pb2.Empty()

    def GetJob(self, request: JobIds, context: grpc.ServicerContext) -> JobSnapshot:
        print(request)
        reader = JobReader(queue)
        jobs = reader.stream_ids(request.job_ids)

        try:
            for snapshot in jobs:
                yield snapshot
        except Exception as e:
            return context.abort(grpc.StatusCode.INTERNAL, "Failed to stream job snapshots")

    def GetUserHistory(self, request: UserId, context: grpc.ServicerContext) -> UserHistory:
        try:
            reader = JobReader(queue)
            jobs = reader.history(request.user_id)

            return jobs
        except Exception as e:
            return context.abort(grpc.StatusCode.INTERNAL, "Failed to read user history")

    def GetNodeDefinitions(self, request: NodeDefRequest, context: grpc.ServicerContext) -> NodeDefs:
        node = NodeDefinition(display_name="whatever", description="whatever", category="none", ux_widgets=[],
                              inputs=[], outputs=[])
        node_defs = NodeDefs(defs={"test": node})

        return node_defs

    def GetModelCatalog(self, request: ModelCatalogRequest, context: grpc.ServicerContext) -> ModelCatalog:
        collection = Collection("models")
        query = collection.collection.where('family', 'in', request.base_family)

        models = {}
        for model in query.stream():
            info = [Models.Info(**info.to_dict()) for info in model.get('info')]
            models[model.key] = Models(info=info)

        return ModelCatalog(models=models)

    def SyncLocalFiles(self, request: empty_pb2.Empty, context: grpc.ServicerContext) -> LocalFiles:
        yield LocalFiles(added=[LocalFile(name="file1.txt")])


def _send_job_request(request: ComfyRequest, context: grpc.ServicerContext):
    job_id = str(uuid4())
    properties = {
        "job_id": job_id,
        "estimated_duration": str(1000),
        "user_uid": context_user_uid.get(),
        "publish_outputs_to_topic": str(True)
    }

    try:
        _add_request_to_queue(request, properties)
    except Exception as e:
        return context.abort(grpc.StatusCode.INTERNAL, "Failed to add job to queue")
    _add_job_to_user_history(context_user_uid.get(), job_id, JobStatus.QUEUED)
    return job_id


def _ensure_context_user(context: grpc.ServicerContext):
    if context_user_uid.get() is None:
        return context.abort(grpc.StatusCode.UNAUTHENTICATED, "Authentication Required")


def _add_request_to_queue(request: ComfyRequest, properties: dict):
    job_topic = make_topic(settings.pulsar.job_queue_namespace, get_affinity(request.workflow))
    queue.publish(job_topic, message=request.SerializeToString(), properties=properties)


def _add_job_to_user_history(user_uid: str, job_id: str, status: JobStatus):
    history_topic = make_topic(settings.pulsar.user_history_namespace, user_uid)
    message = UserHistory.Job(job_id=job_id, status=status).SerializeToString()
    queue.publish(history_topic, message=message, partition_key=job_id)


def _wait_for_job_snapshot(job_id: str):
    topic = make_topic(settings.pulsar.job_snapshot_namespace, job_id)
    listener_name = f"{settings.pulsar.job_snapshot_namespace}-{job_id}"
    consumer = queue.subscribe(
        topic,
        receiver_queue_size=1,
        consumer_name=listener_name,
        subscription_name=listener_name,
        consumer_type=pulsar.ConsumerType.Shared,
    )

    message = consumer.receive()
    return JobSnapshot.FromString(message.data())


def stream_jobs(job_id: str):
    topic = make_topic(settings.pulsar.job_snapshot_namespace, job_id)
    listener_name = f"{settings.pulsar.job_snapshot_namespace}-{job_id}"
    consumer = queue.subscribe(
        topic,
        receiver_queue_size=1,
        consumer_name=listener_name,
        subscription_name=listener_name,
        consumer_type=pulsar.ConsumerType.Shared
    )

    while True:
        try:
            message = consumer.receive()
            yield JobSnapshot.FromString(message.data())
        except Exception as e:
            print(f"Error receiving message: {e}")
            break


def _request_job_cancellation(request: JobId):
    try:
        print(f"Requesting job cancellation: {request.job_id}")
        cancel_topic = make_topic(settings.pulsar.job_cancel_namespace, request.job_id)
        queue.publish(cancel_topic, message=JobSnapshot(job_id=request.job_id).SerializeToString())
    except Exception as e:
        print(f"Error requesting job cancellation: {e}")


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

            if context_user_uid.get() is not None:
                print("Corrupted context var")

            context_user_uid.set(user.uid)
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
