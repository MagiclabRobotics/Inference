from __future__ import annotations

import contextlib
import os
import socket
import time
import traceback
from typing import Any

from loguru import logger
import numpy as np
from openpi.serving import server_logging
from openpi_client import base_policy as _base_policy
from openpi_client import shared_memory_transport as _transport
import torch


class SharedMemoryPolicyServer:
    """Serves a policy over a local shared-memory transport.

    The Unix socket carries only control messages and shared-memory descriptors.
    Array payloads are stored in shared memory blocks.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        socket_path: str = "/tmp/openpi_policy.sock",
        metadata: dict | None = None,
        request_log_every_n: int = 1,
    ) -> None:
        self._policy = policy
        self._socket_path = socket_path
        self._metadata = metadata or {}
        self._request_log_every_n = max(1, int(request_log_every_n))

    def serve_forever(self) -> None:
        parent = os.path.dirname(self._socket_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self._socket_path)

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(self._socket_path)
            server.listen(1)
            logger.info("shared-memory policy server listening socket={}", self._socket_path)
            while True:
                conn, _ = server.accept()
                with conn:
                    self._handle_connection(conn)

    def _handle_connection(self, conn: socket.socket) -> None:
        remote = self._socket_path
        logger.info("shared-memory connection opened socket={}", self._socket_path)
        server_logging.log_server_event("connection_open", transport="shared_memory", remote=remote)
        _transport.send_message(conn, {"type": "metadata", "metadata": self._metadata})

        server_request_id = 0
        while True:
            request_id = None
            request_blocks = []
            response = None
            response_blocks = []
            try:
                start_time = time.monotonic()
                wait_start = time.monotonic()
                message = _transport.recv_message(conn)
                recv_wait_ms = (time.monotonic() - wait_start) * 1000
                if message.get("type") != "infer":
                    raise RuntimeError(f"Unexpected shared-memory request: {message!r}")
                server_request_id += 1
                request_id = message.get("request_id", message.get("request_index", server_request_id))

                decode_start = time.monotonic()
                obs, request_blocks = _transport.decode_payload(
                    message["payload"],
                    copy_arrays=False,
                    unregister_attached=True,
                )
                unpack_ms = (time.monotonic() - decode_start) * 1000

                infer_start = time.monotonic()
                action = self._policy.infer(obs)
                policy_return_ms = (time.monotonic() - infer_start) * 1000
                cuda_sync_start = time.monotonic()
                cuda_synchronized = _synchronize_cuda()
                cuda_sync_ms = (time.monotonic() - cuda_sync_start) * 1000 if cuda_synchronized else None
                infer_ms = (time.monotonic() - infer_start) * 1000
                _transport.close_blocks(request_blocks)
                request_blocks = []

                action["server_timing"] = {
                    "request_id": request_id,
                    "server_request_id": server_request_id,
                    "recv_wait_ms": recv_wait_ms,
                    "unpack_ms": unpack_ms,
                    "infer_ms": infer_ms,
                    "policy_return_ms": policy_return_ms,
                }
                if cuda_sync_ms is not None:
                    action["server_timing"]["cuda_sync_ms"] = cuda_sync_ms

                encode_start = time.monotonic()
                response = _transport.encode_payload(action, prefix="openpi_resp")
                response_blocks = response.blocks
                _transport.unregister_blocks(response_blocks)
                pack_ms = (time.monotonic() - encode_start) * 1000
                send_start = time.monotonic()
                control_bytes = _transport.send_message(
                    conn,
                    {
                        "type": "result",
                        "request_id": request_id,
                        "server_request_id": server_request_id,
                        "payload": response.payload,
                    },
                )
                send_ms = (time.monotonic() - send_start) * 1000
                total_ms = (time.monotonic() - start_time) * 1000

                event = {
                    "transport": "shared_memory",
                    "request_id": request_id,
                    "server_request_id": server_request_id,
                    "remote": remote,
                    "request_shm_bytes": _transport.array_bytes_in_payload(message.get("payload")),
                    "response_shm_bytes": _transport.array_bytes_in_payload(response.payload),
                    "response_control_bytes": control_bytes,
                    "recv_wait_ms": recv_wait_ms,
                    "unpack_ms": unpack_ms,
                    "policy_return_ms": policy_return_ms,
                    "cuda_sync_ms": cuda_sync_ms,
                    "infer_ms": infer_ms,
                    "pack_ms": pack_ms,
                    "send_ms": send_ms,
                    "total_ms": total_ms,
                    "observation": _summarize_value(obs),
                    "action": _summarize_value(action),
                }
                server_logging.log_server_event("inference", **event)
                if server_request_id % self._request_log_every_n == 0:
                    logger.info(
                        "shared-memory inference server_req={} request_id={} wait={:.1f}ms unpack={:.1f}ms infer={:.1f}ms "
                        "policy_return={:.1f}ms cuda_sync={} pack={:.1f}ms send={:.1f}ms total={:.1f}ms "
                        "req_shm={} resp_shm={} control_bytes={}",
                        server_request_id,
                        request_id,
                        recv_wait_ms,
                        unpack_ms,
                        infer_ms,
                        policy_return_ms,
                        _format_optional_ms(cuda_sync_ms),
                        pack_ms,
                        send_ms,
                        total_ms,
                        event["request_shm_bytes"],
                        event["response_shm_bytes"],
                        control_bytes,
                    )
            except EOFError:
                logger.info("shared-memory connection closed socket={} requests={}", self._socket_path, server_request_id)
                server_logging.log_server_event(
                    "connection_close",
                    transport="shared_memory",
                    remote=remote,
                    requests=server_request_id,
                )
                break
            except Exception as exc:
                tb = traceback.format_exc()
                logger.exception(
                    "shared-memory inference failed server_req={} request_id={}: {}",
                    server_request_id,
                    request_id,
                    exc,
                )
                server_logging.log_server_event(
                    "inference_error",
                    transport="shared_memory",
                    request_id=request_id,
                    server_request_id=server_request_id,
                    remote=remote,
                    error=str(exc),
                    traceback=tb,
                )
                _transport.send_message(
                    conn,
                    {
                        "type": "error",
                        "request_id": request_id,
                        "server_request_id": server_request_id,
                        "traceback": tb,
                    },
                )
                break
            finally:
                _transport.close_blocks(request_blocks)
                for block in response_blocks:
                    block.close()


def _synchronize_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    torch.cuda.synchronize()
    return True


def _format_optional_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}ms"


def _summarize_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return type(value).__name__
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _summarize_value(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, list | tuple):
        summary: dict[str, Any] = {"type": type(value).__name__, "len": len(value)}
        if len(value) <= 8:
            summary["items"] = [_summarize_value(item, depth=depth + 1) for item in value]
        return summary
    if isinstance(value, str):
        return {"type": "str", "len": len(value), "preview": value[:120]}
    if isinstance(value, int | float | bool) or value is None:
        return value
    return type(value).__name__
