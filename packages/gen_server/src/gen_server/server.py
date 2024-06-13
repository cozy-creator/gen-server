from contextvars import ContextVar
from concurrent import futures
from typing import Optional

import grpc

from core_extension_1.widgets import WidgetDefinition
from gen_server.globals import CUSTOM_NODES
from gen_server.proto_defs import (
    ComfyGRPCServiceServicer,
    add_ComfyGRPCServiceServicer_to_server,
    NodeDefRequest,
    NodeDefs,
    NodeDefinition,
)
from gen_server.types_1.types_1 import Serializable

user_uid: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


class ComfyServicer(ComfyGRPCServiceServicer):
    def GetNodeDefinitions(
            self, request: NodeDefRequest, context: grpc.ServicerContext
    ) -> NodeDefs:
        defs = _get_node_definitions()
        node_defs = NodeDefs(defs=defs)

        return node_defs


def _get_node_definitions():
    definitions = {}
    for node_type in CUSTOM_NODES:
        node = CUSTOM_NODES[node_type]
        interface = node.update_interface()

        ux_widgets = []
        input_list = []
        output_list = []

        inputs = interface.get("inputs")
        if inputs is not None:
            for name in inputs:
                if isinstance(inputs[name], Serializable):
                    spec = inputs[name].serialize()
                    input_type = spec.get("type")
                else:
                    if isinstance(inputs[name], type):
                        input_type = inputs[name].__name__
                    else:
                        input_type = type(inputs[name]).__name__
                if isinstance(inputs[name], WidgetDefinition):
                    ux_widgets.append(
                        NodeDefinition.UXWidget(widget_type=input_type, spec=spec)
                    )
                else:
                    input_list.append(
                        NodeDefinition.InputDef(edge_type=input_type, display_name=name)
                    )

        outputs = interface.get("outputs")
        if outputs is not None:
            for name in outputs:
                if isinstance(outputs[name], type):
                    output_type = outputs[name].__name__
                else:
                    output_type = type(outputs[name]).__name__
                output_list.append(
                    NodeDefinition.OutputDef(edge_type=output_type, display_name=name)
                )

        definitions[node_type] = NodeDefinition(
            display_name=node_type,
            description=node_type,
            category="none",
            ux_widgets=ux_widgets,
            inputs=input_list,
            outputs=output_list,
        )

    return definitions


def _unary_unary_rpc_terminator(code, details):
    def terminate(ignored_request, context):
        context.abort(code, details)

    return grpc.unary_unary_rpc_method_handler(terminate)


class AuthenticationInterceptor(grpc.ServerInterceptor):
    def __init__(self):
        self._header = "authorization"
        self._terminator = _unary_unary_rpc_terminator(
            grpc.StatusCode.UNAUTHENTICATED, "Invalid auth details"
        )

    def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata)
        if self._header in metadata:
            value = metadata.get(self._header)
            if value is None:
                return self._terminator

            bearer_parts = value.split(maxsplit=1)
            if bearer_parts[0] != "Bearer":
                return self._terminator

            if user_uid.get() is not None:
                print("Corrupted context var")

            return continuation(handler_call_details)
        else:
            return self._terminator


def start_server(host: str, port: int):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=(AuthenticationInterceptor(),),
    )
    add_ComfyGRPCServiceServicer_to_server(ComfyServicer(), server)

    address = f"{host}:{port}"
    server.add_insecure_port(address)
    server.start()

    print(f"GRPC Server started. Listening at {address}")
    server.wait_for_termination()


start_server("0.0.0.0", 50051)
