import asyncio
import json
import os
import sys
import threading
import time
import uuid
from pprint import pprint

import pulsar

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../protobuf')))

from protobuf.grpc_service_v1_pb2 import ComfyRequest, JobSnapshot, UserHistory, JobStatus
from gen_server.job_queue.interface import JobQueue
from gen_server.job_queue.pulsar import Pulsar, make_topic
from gen_server.settings import settings

fake_node = {
    "fake-node-id": {
        "type": "ComfyFakeNode",
        "data": {
            "inputs": {
                "name": "fake name"
            },
            "outputs": {
                "image": "<fake_image_ref>"
            }
        }
    }
}


def is_job_canceled(queue: JobQueue, job_id: str):
    try:
        print("waiting for cancel...")
        cancel_topic = make_topic(settings.pulsar.job_cancel_namespace, job_id)
        cancel_consumer = queue.subscribe(
            cancel_topic,
            receiver_queue_size=1,
            subscription_name=f"cancel-${job_id}",
            consumer_type=pulsar.ConsumerType.Shared
        )

        message = cancel_consumer.receive()
        snapshot = JobSnapshot.FromString(message.data())
        print("requested cancel...")
        return snapshot.job_id == job_id
    except pulsar.ConsumerBusy as e:
        print("Consumer Busy", e)
    except Exception as e:
        print("An unknown error occurred", e)


def cancel_job(queue: JobQueue, user_uid: str, job_id: str):
    try:
        history_topic = make_topic(settings.pulsar.user_history_namespace, user_uid)
        message = UserHistory.Job(job_id=job_id, status=JobStatus.ABORTED).SerializeToString()
        queue.publish(history_topic, message=message, partition_key=job_id)

        print("job aborted")
    except pulsar.ConsumerBusy as e:
        print("Consumer Busy", e)
    except Exception as e:
        print("An unknown error occurred", e)


def delete_cancel_subscription(queue: JobQueue, job_id: str):
    try:
        cancel_topic = make_topic(settings.pulsar.job_cancel_namespace, job_id)
        queue.unsubscribe(cancel_topic, subscription_name=f"cancel-${job_id}")
    except pulsar.ConsumerBusy as e:
        print("Consumer Busy", e)
    except Exception as e:
        print("An unknown error occurred", e)


class Worker:
    def __init__(self, worker_id: str, tenant: str, queue: JobQueue):
        self.worker_id = worker_id
        self.tenant = tenant
        self.queue = queue

    async def start_consumer(self):
        topic = f"persistent://{self.tenant}/{settings.pulsar.job_queue_namespace}/sdxl-1"
        consumer = self.queue.subscribe(
            topic,
            receiver_queue_size=1,
            subscription_name="worker",
            consumer_type=pulsar.ConsumerType.Shared
        )

        while True:
            try:
                msg = consumer.receive()
                print(f"Received message: {msg}")
                self.handle_message(msg, consumer)
            except pulsar.ConsumerBusy as e:
                time.sleep(.1)
                print(f"Consumer busy: {e}... retrying in immediately")
            except Exception as err:
                print(f"Error receiving message: {err}")
                break

    def process_request(self, stop_event, _request: ComfyRequest, callback: callable):
        print("Processing request...")
        (completed, time_taken) = (False, 0)
        complete_threshold = 10
        while not stop_event.is_set():
            print("Job is running...")
            time_taken += .1

            if time_taken >= complete_threshold:
                stop_event.set()
            time.sleep(.1)

        if time_taken >= complete_threshold:
            callback(fake_node)

    def handle_message(self, message, consumer):
        request = ComfyRequest.FromString(message.data())
        properties = message.properties()
        job_id = properties.get('job_id')
        if job_id is None:
            print(f"Job ID not found in request (ID: {request.request_id}) properties")

        is_processing: bool = False
        is_complete: bool = False
        while True:
            # The job is already processing, skip
            if is_processing:
                continue

            # The job is complete, exit the loop
            if is_complete:
                break

            # Start processing the job
            is_processing = True

            # OnComplete callback
            def on_complete(result: dict):
                global is_complete
                is_complete = True
                self.stream_job_results(job_id, request.request_id, result)
                delete_cancel_subscription(self.queue, job_id)

                consumer.acknowledge(message)
                print(f"Job {job_id} is complete")

            # Mock ComfyUI by running the job in a separate thread
            stop_event = threading.Event()
            thread = threading.Thread(target=self.process_request, args=(stop_event, request, on_complete))
            thread.start()

            if is_job_canceled(self.queue, job_id):
                print(f"Job {job_id} was canceled")
                # Mock Interrupt signal to ComfyUI
                stop_event.set()

                consumer.acknowledge(message)
                cancel_job(self.queue, properties['user_uid'], job_id)
                break

    def stream_job_results(self, job_id: str, request_id: str, result: dict):
        topic = f"persistent://{self.tenant}/{settings.pulsar.job_snapshot_namespace}/{job_id}"
        message = JobSnapshot(
            job_id=job_id,
            request_id=request_id,
            snapshot=json.dumps(result).encode()
        ).SerializeToString()

        print(f"Streaming job results to {topic}")
        pprint(result)

        self.queue.publish(topic, message=message)


if __name__ == "__main__":
    worker = Worker(str(uuid.uuid4()), settings.pulsar.tenant, Pulsar(settings.pulsar))
    asyncio.run(worker.start_consumer())
