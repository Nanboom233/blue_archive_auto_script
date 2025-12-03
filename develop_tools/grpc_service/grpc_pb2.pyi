from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

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
    __slots__ = ("timestamp", "level", "message")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    timestamp: int
    level: int
    message: str

    def __init__(self, timestamp: _Optional[int] = ..., level: _Optional[int] = ...,
                 message: _Optional[str] = ...) -> None: ...


class InterfaceInfo(_message.Message):
    __slots__ = ("var_name", "id", "description", "features", "actions")
    VAR_NAME_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    var_name: str
    id: str
    description: str
    features: str
    actions: str

    def __init__(self, var_name: _Optional[str] = ..., id: _Optional[str] = ..., description: _Optional[str] = ...,
                 features: _Optional[str] = ..., actions: _Optional[str] = ...) -> None: ...


class InterfacesCollection(_message.Message):
    __slots__ = ("interfaces",)
    INTERFACES_FIELD_NUMBER: _ClassVar[int]
    interfaces: _containers.RepeatedCompositeFieldContainer[InterfaceInfo]

    def __init__(self, interfaces: _Optional[_Iterable[_Union[InterfaceInfo, _Mapping]]] = ...) -> None: ...
