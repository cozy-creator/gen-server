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

token = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImUyYjIyZmQ0N2VkZTY4MmY2OGZhY2NmZTdjNGNmNWIxMWIxMmI1NGIiLCJ0eXAiOiJKV1QifQ.eyJuYW1lIjoiQWJkdWxyYWhtYW4gWXVzdWYiLCJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20vY2Fwc3VsZXMtZGV2IiwiYXVkIjoiY2Fwc3VsZXMtZGV2IiwiYXV0aF90aW1lIjoxNzE1MzcyMjQwLCJ1c2VyX2lkIjoiVDFTRHpaenRnUGdIUUtjR1JPZTZCRnF5OWpKMiIsInN1YiI6IlQxU0R6Wnp0Z1BnSFFLY0dST2U2QkZxeTlqSjIiLCJpYXQiOjE3MTYwNjE5NDksImV4cCI6MTcxNjA2NTU0OSwiZW1haWwiOiJhYmR1bHJhaG1hbnl1c3VmMTI1QGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJmaXJlYmFzZSI6eyJpZGVudGl0aWVzIjp7ImFwcGxlLmNvbSI6WyIwMDA5MTYuZmM1OGYwMDI0Y2RlNGViYzk2MWU0OGY0MjY4ZGUyNWIuMTUzNCJdLCJnb29nbGUuY29tIjpbIjExMzAzMjc2NDEzMzY1NjAxNTk3MCJdLCJlbWFpbCI6WyJhYmR1bHJhaG1hbnl1c3VmMTI1QGdtYWlsLmNvbSJdfSwic2lnbl9pbl9wcm92aWRlciI6Imdvb2dsZS5jb20ifX0.jc2f_Kpi-0CxOOodVdNvoURkURdCr5HKOfEq-CZK_eIwW8Di_vi29s1HDOvrAZnskshwmaWEytSvHAFbSqbn0d4JSNA9AOEpDOL1pLMl0_JhFtfO68OPxVaTqdAHVNEMGNuLR4kXo19PWsmtUaqUC11Y2-H1pac-dxsPam-frNTbB0zOoIexMOkxqjXeEkS7Je1Y6wixdZn82b_-E1pMsq8-PczrMufqwYf4mrvNE_M520nkl1z-Wut0Ggv6ZJXfwhnthqUcu8u5ySZ5AwAi_-c7GhZ1C9AKEn_Ah0sSOZICa195QQmejX60r5KajG0c56gpzR3O_1F01mE8MsaETw"
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
    # run_job_sync()
    # get_user_history()
    # get_job()
    # time.sleep(2)
    # print("Cancelling job...", job.job_id)
    # cancel_job(job.job_id)
