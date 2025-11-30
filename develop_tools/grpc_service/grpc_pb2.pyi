from typing import ClassVar as _ClassVar, Optional as _Optional

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message

DESCRIPTOR: _descriptor.FileDescriptor


class CommandRequest(_message.Message):
    __slots__ = ("command",)
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    command: str

    def __init__(self, command: _Optional[str] = ...) -> None: ...


class CommandResponse(_message.Message):
    __slots__ = ("success", "message", "timestamp")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    timestamp: int

    def __init__(self, success: bool = ..., message: _Optional[str] = ..., timestamp: _Optional[int] = ...) -> None: ...


class HeartbeatPacket(_message.Message):
    __slots__ = ("timestamp",)
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    timestamp: int

    def __init__(self, timestamp: _Optional[int] = ...) -> None: ...


class LogEntry(_message.Message):
    __slots__ = ("timestamp", "message")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    timestamp: int
    message: str

    def __init__(self, timestamp: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...
