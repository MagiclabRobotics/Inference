"""Shared-memory transport helpers for local policy serving.

Large NumPy arrays are placed in POSIX shared memory blocks. The control
channel only carries small JSON descriptors over a local Unix domain socket.
"""

from __future__ import annotations

import json
from multiprocessing import resource_tracker
from multiprocessing import shared_memory
import socket
import struct
from typing import Any
import uuid

import numpy as np


_HEADER = struct.Struct("!Q")
_ARRAY_MARKER = "__openpi_shared_memory_ndarray__"
_TUPLE_MARKER = "__openpi_tuple__"
_BYTES_MARKER = "__openpi_bytes__"


class SharedMemoryPayload:
    """Encoded payload and the shared-memory blocks created for it."""

    def __init__(self, payload: Any, blocks: list[shared_memory.SharedMemory]):
        self.payload = payload
        self.blocks = blocks

    def close(self) -> None:
        for block in self.blocks:
            block.close()

    def unlink(self) -> None:
        for block in self.blocks:
            try:
                block.unlink()
            except FileNotFoundError:
                pass

    def close_and_unlink(self) -> None:
        self.close()
        self.unlink()


def encode_payload(value: Any, *, prefix: str = "openpi") -> SharedMemoryPayload:
    blocks: list[shared_memory.SharedMemory] = []
    payload = _encode_value(value, blocks, prefix=prefix)
    return SharedMemoryPayload(payload, blocks)


def decode_payload(
    value: Any,
    *,
    copy_arrays: bool,
    unlink_arrays: bool = False,
    unregister_attached: bool = False,
) -> tuple[Any, list[shared_memory.SharedMemory]]:
    blocks: list[shared_memory.SharedMemory] = []
    decoded = _decode_value(
        value,
        blocks,
        copy_arrays=copy_arrays,
        unregister_attached=unregister_attached,
    )
    if unlink_arrays:
        for block in blocks:
            try:
                block.unlink()
            except FileNotFoundError:
                pass
    return decoded, blocks


def close_blocks(blocks: list[shared_memory.SharedMemory]) -> None:
    for block in blocks:
        block.close()


def unregister_blocks(blocks: list[shared_memory.SharedMemory]) -> None:
    for block in blocks:
        _unregister_attached_block(block)


def unlink_blocks(blocks: list[shared_memory.SharedMemory]) -> None:
    for block in blocks:
        try:
            block.unlink()
        except FileNotFoundError:
            pass


def close_and_unlink_blocks(blocks: list[shared_memory.SharedMemory]) -> None:
    close_blocks(blocks)
    unlink_blocks(blocks)


def send_message(sock: socket.socket, message: dict[str, Any]) -> int:
    data = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sock.sendall(_HEADER.pack(len(data)))
    sock.sendall(data)
    return len(data)


def recv_message(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, _HEADER.size)
    if header is None:
        raise EOFError("socket closed")
    size = _HEADER.unpack(header)[0]
    if size <= 0:
        raise ValueError(f"Invalid message size: {size}")
    data = _recv_exact(sock, size)
    if data is None:
        raise EOFError("socket closed while reading message")
    message = json.loads(data.decode("utf-8"))
    if not isinstance(message, dict):
        raise ValueError(f"Expected JSON object message, got {type(message).__name__}")
    return message


def array_bytes_in_payload(value: Any) -> int:
    if isinstance(value, dict):
        if value.get(_ARRAY_MARKER) is True:
            return int(value["nbytes"])
        return sum(array_bytes_in_payload(item) for item in value.values())
    if isinstance(value, list):
        return sum(array_bytes_in_payload(item) for item in value)
    return 0


def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _encode_value(value: Any, blocks: list[shared_memory.SharedMemory], *, prefix: str) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        name = f"{prefix}_{uuid.uuid4().hex}"
        nbytes = int(array.nbytes)
        block = shared_memory.SharedMemory(name=name, create=True, size=max(1, nbytes))
        if nbytes:
            target = np.ndarray(array.shape, dtype=array.dtype, buffer=block.buf)
            target[...] = array
        blocks.append(block)
        return {
            _ARRAY_MARKER: True,
            "name": name,
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "nbytes": nbytes,
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _encode_value(item, blocks, prefix=prefix) for key, item in value.items()}
    if isinstance(value, tuple):
        return {_TUPLE_MARKER: [_encode_value(item, blocks, prefix=prefix) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item, blocks, prefix=prefix) for item in value]
    if isinstance(value, bytes):
        return {_BYTES_MARKER: list(value)}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise TypeError(f"Unsupported shared-memory payload value: {type(value).__name__}")


def _decode_value(
    value: Any,
    blocks: list[shared_memory.SharedMemory],
    *,
    copy_arrays: bool,
    unregister_attached: bool,
) -> Any:
    if isinstance(value, dict):
        if value.get(_ARRAY_MARKER) is True:
            name = str(value["name"])
            block = shared_memory.SharedMemory(name=name)
            if unregister_attached:
                _unregister_attached_block(block)
            blocks.append(block)
            array = np.ndarray(
                tuple(int(dim) for dim in value["shape"]),
                dtype=np.dtype(str(value["dtype"])),
                buffer=block.buf,
            )
            return array.copy() if copy_arrays else array
        if _TUPLE_MARKER in value:
            return tuple(
                _decode_value(
                    item,
                    blocks,
                    copy_arrays=copy_arrays,
                    unregister_attached=unregister_attached,
                )
                for item in value[_TUPLE_MARKER]
            )
        if _BYTES_MARKER in value:
            return bytes(value[_BYTES_MARKER])
        return {
            key: _decode_value(
                item,
                blocks,
                copy_arrays=copy_arrays,
                unregister_attached=unregister_attached,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _decode_value(
                item,
                blocks,
                copy_arrays=copy_arrays,
                unregister_attached=unregister_attached,
            )
            for item in value
        ]
    return value


def _unregister_attached_block(block: shared_memory.SharedMemory) -> None:
    # SharedMemory(name=...) registers the segment with resource_tracker even when
    # this process does not own it. The creator/unlinker owns cleanup, so prevent
    # spurious unlink attempts and shutdown warnings from attached processes.
    try:
        resource_tracker.unregister(block._name, "shared_memory")  # noqa: SLF001
    except Exception:
        pass
