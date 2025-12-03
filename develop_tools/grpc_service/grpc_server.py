import enum
import json
import os
import socket
import tempfile
import time
import traceback
from concurrent import futures
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import grpc
from rich.console import Console
from rich.markup import escape

import core.navigator.interfaces as INTERFACES
from . import grpc_pb2
from . import grpc_pb2_grpc

console = Console()

if TYPE_CHECKING:
    from core.Baas_thread import Baas_thread


class GrpcServicer(grpc_pb2_grpc.GrpcServiceServicer):
    baas_thread: Baas_thread
    emitter: GrpcServicer.LoggerSignalEmitter

    class LoggerSignalEmitter:
        """
        A logger_signal-compatible emitter with an emit(message: str) method.
        It can forward messages to client subscribers and/or print to server stdout
        according to the selected mode. It does not depend on utils.Logger internals.
        """
        log_queue: list[grpc_pb2.LogEntry]

        class OutputMode(enum.Enum):
            SERVER_ONLY = 0
            CLIENT_ONLY = 1
            BOTH = 2

        def __init__(self, output_mode: OutputMode):
            self._mode = output_mode
            self.log_queue = []

        def emit(self, level: int, message: str):
            # message is what utils.Logger passes (could be HTML text)
            if self._mode in (GrpcServicer.LoggerSignalEmitter.OutputMode.SERVER_ONLY,
                              GrpcServicer.LoggerSignalEmitter.OutputMode.BOTH):
                levels_str = ["INFO", "WARNING", "ERROR", "CRITICAL"]
                levels_color = ["#2d8cf0", "#ff9900", "#ed3f14", "#3e0480"]
                console.print(
                    f'[HANDLED][{levels_color[level - 1]}]{levels_str[level - 1]} | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | {escape(message)}[/]',
                    soft_wrap=True)
            if self._mode in (GrpcServicer.LoggerSignalEmitter.OutputMode.CLIENT_ONLY,
                              GrpcServicer.LoggerSignalEmitter.OutputMode.BOTH):
                self.log_queue.append(
                    grpc_pb2.LogEntry(timestamp=int(time.time_ns() // 1_000_000), level=level, message=message))

    def __init__(self, baas_thread: Baas_thread, logger_emitter: LoggerSignalEmitter):
        self.baas_thread = baas_thread
        self.emitter = logger_emitter
        self.log_queue = []

    def Command(self, request, context):
        received_cmd = request.command
        print(f"Receive Command: {received_cmd}")
        try:
            return grpc_pb2.CommandResponse(
                success=True,
                message=str(eval(received_cmd)),
                timestamp=int(time.time())
            )
        except Exception as e:
            return grpc_pb2.CommandResponse(
                success=False,
                message=f"Command encountered an error: {e}, traceback: {traceback.format_exc()}",
                timestamp=int(time.time())
            )

    def Heartbeat(self, request, context):
        return grpc_pb2.HeartbeatPacket(
            timestamp=time.time_ns() // 1_000_000  # return system time in milliseconds
        )

    def SubscribeLogs(self, request, context):
        try:
            # Continuously stream while the RPC is active
            while context.is_active():
                while self.emitter.log_queue:
                    yield self.emitter.log_queue.pop(0)
                time.sleep(0.05)
        finally:
            pass

    def GetInterfaces(self, request, context):
        try:
            interfaces: list[dict] = INTERFACES.InterfacesCSTEditor().read_interfaces()
            grpc_interface_infos: list[grpc_pb2.InterfaceInfo] = []
            for iface in interfaces:
                grpc_interface_infos.append(
                    grpc_pb2.InterfaceInfo(
                        var_name=iface["var_name"],
                        id=iface["id"],
                        description=iface["description"],
                        features=json.dumps(iface["features"], indent=4),
                        actions=json.dumps(iface["actions"], indent=4)
                    )
                )
            return grpc_pb2.InterfacesCollection(interfaces=grpc_interface_infos)
        except Exception as e:
            context.set_details(f"GetInterfaces encountered an error: {e}, traceback: {traceback.format_exc()}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return grpc_pb2.InterfacesCollection()

    def WriteInterfaces(self, request, context):
        interfaces: list[dict] = []
        for grpc_iface in request.interfaces:
            interfaces.append(
                {
                    "var_name": grpc_iface.var_name,
                    "id": grpc_iface.id,
                    "description": grpc_iface.description,
                    "features": json.loads(grpc_iface.features),
                    "actions": json.loads(grpc_iface.actions)
                }
            )
        new_code = INTERFACES.InterfacesCSTEditor().generate_code(interfaces)
        file_path = Path(INTERFACES.InterfacesCSTEditor().get_interfaces_file_path())
        fd, temp_file_path = tempfile.mkstemp(prefix=file_path.name + ".", suffix=".tmp", dir=file_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(new_code)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file_path, file_path)
            return grpc_pb2.CommandResponse(
                success=True,
                message="Interfaces written successfully.",
                timestamp=int(time.time_ns() // 1_000_000)
            )
        except Exception as e:
            try:
                os.remove(temp_file_path)
            except FileNotFoundError:
                # ignored
                pass
            return grpc_pb2.CommandResponse(
                success=False,
                message=f"Interfaces written encountered an error: {e}, traceback: {traceback.format_exc()}",
                timestamp=int(time.time_ns() // 1_000_000)
            )


def find_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_server(baas_thread: Baas_thread, port: int | None = None,
                 logger_emitter: Optional[GrpcServicer.LoggerSignalEmitter] = None, ):
    port = port or find_free_port()

    # Use the nested LoggerSignalEmitter and OutputMode definitions on GrpcServicer
    emitter = logger_emitter or GrpcServicer.LoggerSignalEmitter(
        output_mode=GrpcServicer.LoggerSignalEmitter.OutputMode.SERVER_ONLY
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8), options=[
        ("grpc.so_reuseport", 0),
        ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ])
    grpc_pb2_grpc.add_GrpcServiceServicer_to_server(GrpcServicer(baas_thread, emitter), server)
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    print(f"starting server on port {port}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        pass
