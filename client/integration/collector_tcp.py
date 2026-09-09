"""通过 piperserver-master_xh 推理报文口查询关节（与采集进程解耦的 IPC）。"""

from __future__ import annotations

import json
import socket
import threading
from typing import Any

import numpy as np

from integration.collector_contract import ACTION_DIM


class CollectorTcpClient:
    """一行 JSON 请求/响应，对接 xh ``InferencePacketServer`` TCP（默认 9000）。"""

    def __init__(self, host: str, port: int, *, name: str = "collector_tcp"):
        self.host = str(host)
        self.port = int(port)
        self.name = name
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None

    def _ensure_connected(self) -> None:
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((self.host, self.port))
        sock.settimeout(float(self._request_timeout_s))
        self._sock = sock

    _request_timeout_s = 5.0

    def _request(self, payload: dict) -> dict:
        with self._lock:
            self._ensure_connected()
            line = json.dumps(payload, ensure_ascii=False) + "\n"
            self._sock.sendall(line.encode("utf-8"))
            buf = ""
            while "\n" not in buf:
                chunk = self._sock.recv(65536)
                if not chunk:
                    raise ConnectionError(f"[{self.name}] 采集连接断开")
                buf += chunk.decode("utf-8")
            return json.loads(buf.splitlines()[0])

    def get_joint_state_at_or_after(self, frame_time: float) -> dict[str, Any]:
        resp = self._request(
            {"cmd": "get_joint_state", "params": {"after_stamp": float(frame_time)}}
        )
        if not resp.get("status"):
            raise RuntimeError(f"[{self.name}] get_joint_state 失败: {resp.get('message')}")
        data = resp.get("data") or {}
        qpos = np.asarray(data.get("qpos"), dtype=np.float32).reshape(-1)
        if qpos.size != ACTION_DIM:
            raise ValueError(f"qpos 必须是 {ACTION_DIM} 维，收到 {qpos.size}")
        state_ts = float(data.get("state_timestamp", data.get("left_stamp_sec", frame_time)))
        return {
            "qpos": qpos,
            "state_timestamp": state_ts,
            "sync_frame_time": float(frame_time),
            "left_stamp_sec": data.get("left_stamp_sec"),
            "right_stamp_sec": data.get("right_stamp_sec"),
        }
