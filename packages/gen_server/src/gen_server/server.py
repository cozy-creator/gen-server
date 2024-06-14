from contextvars import ContextVar
from concurrent import futures
from typing import Optional, Type, Any

import grpc

from core_extension_1.widgets import WidgetDefinition
from gen_server.globals import CUSTOM_NODES
from gen_server.proto_defs import (
    ComfyGRPCServiceServicer,
    add_ComfyGRPCServiceServicer_to_server,
    NodeDefRequest,
    NodeDefs,
    NodeDefinition,
    ComfyRequest,
)
from gen_server.types_1.types_1 import Serializable
from gen_server.workflow.runner import WorkflowRunner


user_uid: ContextVar[Optional[str]] = ContextVar("user_id", default=None)

# load_extensions(
#     "entry_point",
#     {"LoadCheckpoint": LoadCheckpoint, "CreatePipe": CreatePipe, "RunPipe": RunPipe},
# )


class ComfyServicer(ComfyGRPCServiceServicer):
    def Run(self, request: ComfyRequest, context: grpc.ServicerContext):
        _handle_run_workflow(request, context)

    def GetNodeDefinitions(
        self, _request: NodeDefRequest, _context: grpc.ServicerContext
    ) -> NodeDefs:
        defs = _get_node_definitions()
        node_defs = NodeDefs(defs=defs)
        print(node_defs)

        return node_defs


def class_type(cls: Type[Any]):
    package_name = cls.__module__.split(".")[0]
    return f"{package_name}.{cls.__name__}"


async def _handle_run_workflow(request: ComfyRequest, _context: grpc.ServicerContext):
    runner = WorkflowRunner(request.workflow, {})
    result = await runner.run()

    return result


def _get_node_definitions():
    definitions = {}
    for key, node_type in CUSTOM_NODES.items():
        node = CUSTOM_NODES[node_type]
        interface = node.update_interface()

        input_list = []
        output_list = []
        ux_widget_list = []

        inputs = interface.get("inputs")
        if inputs is not None:
            for name in inputs:
                if isinstance(inputs[name], Serializable):
                    spec = inputs[name].serialize()
                    input_type = spec.get("type")
                else:
                    if isinstance(inputs[name], type):
                        input_type = class_type(inputs[name])
                    else:
                        input_type = class_type(type(inputs[name]))
                if isinstance(inputs[name], WidgetDefinition):
                    ux_widget_list.append(
                        NodeDefinition.UXWidget(spec=spec, widget_type=input_type)
                    )
                else:
                    input_list.append(
                        NodeDefinition.InputDef(
                            display_name=name, edge_type=input_type, required=True
                        )
                    )

        outputs = interface.get("outputs")
        if outputs is not None:
            for name in outputs:
                if isinstance(outputs[name], type):
                    output_type = class_type(outputs[name])
                else:
                    output_type = class_type(type(outputs[name]))

                output_list.append(
                    NodeDefinition.InputDef(display_name=name, edge_type=output_type)
                )

        definitions[node_type] = NodeDefinition(
            display_name=node.name,
            description=node.description,
            category=node.category,
            inputs=input_list,
            outputs=output_list,
            ux_widgets=ux_widget_list,
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
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_ComfyGRPCServiceServicer_to_server(ComfyServicer(), server)

    address = f"{host}:{port}"
    server.add_insecure_port(address)
    server.start()

    print(f"GRPC Server started. Listening at {address}")
    server.wait_for_termination()


start_server("0.0.0.0", 50051)
