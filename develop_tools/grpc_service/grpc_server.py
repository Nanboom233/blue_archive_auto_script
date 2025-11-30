import enum
import socket
import time
from concurrent import futures
from typing import TYPE_CHECKING, Optional

import grpc

from . import grpc_pb2
from . import grpc_pb2_grpc

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

        def emit(self, message: str):
            # message is what utils.Logger passes (could be HTML text)
            if self._mode in (GrpcServicer.LoggerSignalEmitter.OutputMode.SERVER_ONLY,
                              GrpcServicer.LoggerSignalEmitter.OutputMode.BOTH):
                try:
                    print(message)
                except Exception:
                    # ignored
                    pass
            if self._mode in (GrpcServicer.LoggerSignalEmitter.OutputMode.CLIENT_ONLY,
                              GrpcServicer.LoggerSignalEmitter.OutputMode.BOTH):
                self.log_queue.append(grpc_pb2.LogEntry(timestamp=int(time.time_ns() // 1_000_000), message=message))

    def __init__(self, baas_thread: Baas_thread, logger_emitter: LoggerSignalEmitter):
        self.baas_thread = baas_thread
        self.emitter = logger_emitter
        self.log_queue = []

    def command(self, request, context):
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
                message=str(e),
                timestamp=int(time.time())
            )

    def heartbeat(self, request, context):
        return grpc_pb2.HeartbeatPacket(
            timestamp=time.time_ns() // 1_000_000  # return system time in milliseconds
        )

    def subscribe_logs(self, request, context):
        print("logs subscription started")
        try:
            # Continuously stream while the RPC is active
            while context.is_active():
                while self.emitter.log_queue:
                    yield self.emitter.log_queue.pop(0)
                time.sleep(0.05)
        finally:
            pass


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
