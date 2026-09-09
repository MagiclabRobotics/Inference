import itertools
import logging
import socket
import time
from typing import Dict, Optional

from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client import shared_memory_transport as _transport


class SharedMemoryClientPolicy(_base_policy.BasePolicy):
    """Policy client that exchanges NumPy arrays through shared memory.

    A Unix domain socket is used only for small control messages and shared
    memory descriptors. This transport works only when client and server run on
    the same host.
    """

    def __init__(
        self,
        socket_path: str = "/tmp/openpi_policy.sock",
        *,
        connect_timeout_s: Optional[float] = None,
        retry_interval_s: float = 1.0,
    ) -> None:
        self._socket_path = socket_path
        self._connect_timeout_s = connect_timeout_s
        self._retry_interval_s = float(retry_interval_s)
        self._request_counter = itertools.count(1)
        self._sock, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def close(self) -> None:
        self._sock.close()

    def _wait_for_server(self) -> tuple[socket.socket, Dict]:
        logging.info("Waiting for shared-memory policy server at %s...", self._socket_path)
        deadline = None if self._connect_timeout_s is None else time.monotonic() + float(self._connect_timeout_s)
        while True:
            sock = None
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(self._socket_path)
                message = _transport.recv_message(sock)
                if message.get("type") != "metadata":
                    raise RuntimeError(f"Unexpected server handshake: {message!r}")
                metadata = message.get("metadata") or {}
                if not isinstance(metadata, dict):
                    raise RuntimeError(f"Expected metadata dict, got {type(metadata).__name__}")
                return sock, metadata
            except (ConnectionRefusedError, EOFError, FileNotFoundError, OSError):
                if sock is not None:
                    sock.close()
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for shared-memory server: {self._socket_path}") from None
                logging.info("Still waiting for shared-memory policy server...")
                time.sleep(self._retry_interval_s)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        request_id = next(self._request_counter)
        roundtrip_start = time.monotonic()
        encode_start = time.monotonic()
        request = _transport.encode_payload(obs, prefix="openpi_req")
        encode_ms = (time.monotonic() - encode_start) * 1000
        request_shm_bytes = _transport.array_bytes_in_payload(request.payload)
        try:
            logging.debug("Sending shared-memory inference request_id=%s", request_id)
            send_start = time.monotonic()
            request_control_bytes = _transport.send_message(
                self._sock,
                {
                    "type": "infer",
                    "request_id": request_id,
                    "payload": request.payload,
                },
            )
            send_ms = (time.monotonic() - send_start) * 1000
            wait_start = time.monotonic()
            message = _transport.recv_message(self._sock)
            wait_response_ms = (time.monotonic() - wait_start) * 1000
        finally:
            request.close_and_unlink()

        if message.get("type") == "error":
            raise RuntimeError(
                f"Error in shared-memory inference server for request_id={message.get('request_id', message.get('request_index'))}:\n"
                f"{message.get('traceback', '')}"
            )
        if message.get("type") != "result":
            raise RuntimeError(f"Unexpected shared-memory server response: {message!r}")
        response_id = message.get("request_id", message.get("request_index"))
        if response_id != request_id:
            logging.warning(
                "Shared-memory inference request id mismatch: sent=%s received=%s server_request_id=%s",
                request_id,
                response_id,
                message.get("server_request_id"),
            )
        logging.debug(
            "Received shared-memory inference response request_id=%s server_request_id=%s",
            response_id,
            message.get("server_request_id"),
        )

        response_blocks = []
        try:
            decode_start = time.monotonic()
            result, response_blocks = _transport.decode_payload(
                message["payload"],
                copy_arrays=True,
                unlink_arrays=True,
            )
            decode_ms = (time.monotonic() - decode_start) * 1000
            roundtrip_ms = (time.monotonic() - roundtrip_start) * 1000
            if not isinstance(result, dict):
                raise RuntimeError(f"Expected inference result dict, got {type(result).__name__}")
            result["client_timing"] = {
                "transport": "shared_memory",
                "request_id": request_id,
                "encode_ms": encode_ms,
                "send_ms": send_ms,
                "wait_response_ms": wait_response_ms,
                "unpack_ms": decode_ms,
                "roundtrip_ms": roundtrip_ms,
                "request_shm_bytes": request_shm_bytes,
                "request_control_bytes": request_control_bytes,
                "response_shm_bytes": _transport.array_bytes_in_payload(message.get("payload")),
            }
            logging.debug(
                "Received shared-memory inference response request_id=%s server_request_id=%s "
                "encode=%.3fms send=%.3fms wait_response=%.3fms unpack=%.3fms roundtrip=%.3fms "
                "request_shm_bytes=%s request_control_bytes=%s response_shm_bytes=%s",
                response_id,
                message.get("server_request_id"),
                encode_ms,
                send_ms,
                wait_response_ms,
                decode_ms,
                roundtrip_ms,
                request_shm_bytes,
                request_control_bytes,
                _transport.array_bytes_in_payload(message.get("payload")),
            )
            return result
        finally:
            _transport.close_blocks(response_blocks)

    @override
    def reset(self) -> None:
        pass
