import os
import sys
import time
import uuid
import unittest
from pprint import pprint

import grpc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../protobuf')))

from protobuf.grpc_service_v1_pb2 import ComfyRequest, UserId, JobIds, JobId
from protobuf.grpc_service_v1_pb2_grpc import ComfyGRPCServiceStub

token = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImUyYjIyZmQ0N2VkZTY4MmY2OGZhY2NmZTdjNGNmNWIxMWIxMmI1NGIiLCJ0eXAiOiJKV1QifQ.eyJuYW1lIjoiQWJkdWxyYWhtYW4gWXVzdWYiLCJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20vY2Fwc3VsZXMtZGV2IiwiYXVkIjoiY2Fwc3VsZXMtZGV2IiwiYXV0aF90aW1lIjoxNzE1MzcyMjQwLCJ1c2VyX2lkIjoiVDFTRHpaenRnUGdIUUtjR1JPZTZCRnF5OWpKMiIsInN1YiI6IlQxU0R6Wnp0Z1BnSFFLY0dST2U2QkZxeTlqSjIiLCJpYXQiOjE3MTU5NjY1NzIsImV4cCI6MTcxNTk3MDE3MiwiZW1haWwiOiJhYmR1bHJhaG1hbnl1c3VmMTI1QGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJmaXJlYmFzZSI6eyJpZGVudGl0aWVzIjp7ImFwcGxlLmNvbSI6WyIwMDA5MTYuZmM1OGYwMDI0Y2RlNGViYzk2MWU0OGY0MjY4ZGUyNWIuMTUzNCJdLCJnb29nbGUuY29tIjpbIjExMzAzMjc2NDEzMzY1NjAxNTk3MCJdLCJlbWFpbCI6WyJhYmR1bHJhaG1hbnl1c3VmMTI1QGdtYWlsLmNvbSJdfSwic2lnbl9pbl9wcm92aWRlciI6Imdvb2dsZS5jb20ifX0.EirAy1yCqWO3kuGlUHs8_6EnqswtiUma38Y7JyAD_g-MHNszeqtTeQ0wtpjNsfVcDHNN8Ghw4Ff_YSUFwy-q9bIWsIEOa8zPGx7QuZwEimp5K27hJQBwzjfyY8cJqoQB6REcQ-jaWBvEejlP-WvAKbZR7lwpsLTtYbVtfW4aNTJwzd3t99SuHfapx_StNkbx3W3TZWXFUZlwobwJFQdKkDcBgsi2YoRGYP_vzGRgbrQR0dciDXhhRpbZhhQDmpcvIfopX5436ca0z8J0XWDbRcCsIZ7dcarU0PvKfotL6WAWrLPzL0JyhvwF4mu8ztyHwlNftAa3deKl2IwMPJIXtg"
auth_metadata = ("authorization", f"Bearer {token}")


def create_channel():
    yield grpc.insecure_channel("localhost:50051")


def run_job():
    with grpc.insecure_channel("localhost:50051") as channel:
        comfy_stub = ComfyGRPCServiceStub(channel)
        request = ComfyRequest(request_id=str(uuid.uuid4()), serialized_graph=None)

        try:
            response = comfy_stub.Run.with_call(request, metadata=[auth_metadata])
            print("----- Run job -----")
            return response[0]
        except grpc.RpcError as rpc_error:
            print("Received error: %s", rpc_error)
            return rpc_error


def run_job_sync():
    with grpc.insecure_channel("localhost:50051") as channel:
        comfy_stub = ComfyGRPCServiceStub(channel)
        request = ComfyRequest(request_id=str(uuid.uuid4()), serialized_graph=None)

        try:
            response = comfy_stub.RunSync.with_call(request, metadata=[auth_metadata])
            print("----- Run job sync -----")
            pprint(response)
            return response
        except grpc.RpcError as rpc_error:
            print("Received error: %s", rpc_error)
            return rpc_error


def get_user_history():
    with grpc.insecure_channel("localhost:50051") as channel:
        comfy_stub = ComfyGRPCServiceStub(channel)
        request = UserId(user_id="T1SDzZztgPgHQKcGROe6BFqy9jJ2")

        try:
            response = comfy_stub.GetUserHistory.with_call(request, metadata=[auth_metadata])
            print("----- Received user history -----")
            pprint(response)
            return response
        except grpc.RpcError as rpc_error:
            print("Received error: %s", rpc_error)
            return rpc_error


def get_job():
    with grpc.insecure_channel("localhost:50051") as channel:
        comfy_stub = ComfyGRPCServiceStub(channel)
        request = JobIds(job_ids=["3c5540a7-33f1-4efa-8b8c-0c6673bc6f9d"])

        try:
            response = comfy_stub.GetJob(request, metadata=[auth_metadata])
            print(response)
            # for snapshot in response:
            #     print("----- Received snapshot -----")
            #     pprint(snapshot)
            #     yield snapshot
        except Exception as rpc_error:
            print("Received error: %s", rpc_error)
            return rpc_error


def cancel_job(job_id: str):
    with grpc.insecure_channel("localhost:50051") as channel:
        comfy_stub = ComfyGRPCServiceStub(channel)
        request = JobId(job_id=job_id)

        try:
            response = comfy_stub.CancelJob.with_call(request, metadata=[auth_metadata])
            print("----- Cancel job -----")
            pprint(response)
            return response
        except grpc.RpcError as rpc_error:
            print("Received error: %s", rpc_error)
            return rpc_error


if __name__ == '__main__':
    # job = run_job()
    run_job_sync()
    # get_user_history()
    # get_job()

    # time.sleep(.2)
    # print("Cancelling job...", job.job_id)
    # cancel_job(job.job_id)
