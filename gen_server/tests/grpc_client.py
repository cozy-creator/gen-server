import os
import sys
import uuid

import grpc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../protobuf')))

from protobuf.grpc_service_v1_pb2 import ComfyRequest
from protobuf.grpc_service_v1_pb2_grpc import ComfyGRPCServiceStub

token = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImUyYjIyZmQ0N2VkZTY4MmY2OGZhY2NmZTdjNGNmNWIxMWIxMmI1NGIiLCJ0eXAiOiJKV1QifQ.eyJuYW1lIjoiQWJkdWxyYWhtYW4gWXVzdWYiLCJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20vY2Fwc3VsZXMtZGV2IiwiYXVkIjoiY2Fwc3VsZXMtZGV2IiwiYXV0aF90aW1lIjoxNzE1MzcyMjQwLCJ1c2VyX2lkIjoiVDFTRHpaenRnUGdIUUtjR1JPZTZCRnF5OWpKMiIsInN1YiI6IlQxU0R6Wnp0Z1BnSFFLY0dST2U2QkZxeTlqSjIiLCJpYXQiOjE3MTU0NDQ2NTUsImV4cCI6MTcxNTQ0ODI1NSwiZW1haWwiOiJhYmR1bHJhaG1hbnl1c3VmMTI1QGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJmaXJlYmFzZSI6eyJpZGVudGl0aWVzIjp7ImFwcGxlLmNvbSI6WyIwMDA5MTYuZmM1OGYwMDI0Y2RlNGViYzk2MWU0OGY0MjY4ZGUyNWIuMTUzNCJdLCJnb29nbGUuY29tIjpbIjExMzAzMjc2NDEzMzY1NjAxNTk3MCJdLCJlbWFpbCI6WyJhYmR1bHJhaG1hbnl1c3VmMTI1QGdtYWlsLmNvbSJdfSwic2lnbl9pbl9wcm92aWRlciI6Imdvb2dsZS5jb20ifX0.UJElpLQA0pwrFhCAnN8vaiQ17NxJiuu4x7eLsJo2BfifHsx79DWmqMzruasq4d6TDp3Cjowh5VO1RZIbfWQfmX9evkHMCPa2kJGn4JHB-DvF5BT2AzVCpVZB8-VTh3JXUB83WfaW4Zv0jQrlNHCwxA6Xy6-ugmDyxFKtmPGI1-XFuFvosEgQ54DcN-qruUttRTJJqYxyCEHWhoPzErxIN_tjfgwhx0R2EyAJy6qkj2EHYXUJdmtRL2HJlYcwKnoKxqd_CwlFPm9UBiaIa1CHIwTGBOuiNVu3vJV3yglTXVyy7hLI8ev8BG2OJcEEGu0xYt0UNuzkH0u5PO-_VtiXgA"
auth_metadata = ("authorization", f"Bearer {token}")


def create_channel():
    yield grpc.insecure_channel("localhost:50051")


def run_job():
    with grpc.insecure_channel("localhost:50051") as channel:
        comfy_stub = ComfyGRPCServiceStub(channel)
        request = ComfyRequest(request_id=str(uuid.uuid4()), serialized_graph=None)

        try:
            response = comfy_stub.Run.with_call(request, metadata=[("authorization", f"Bearer {token}")])
        except grpc.RpcError as rpc_error:
            print("Received error: %s", rpc_error)
            return rpc_error
        else:
            print("Received message: %s", response)
            return response


if __name__ == "__main__":
    run_job()
