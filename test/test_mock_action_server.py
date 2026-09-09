from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_CLIENT_SRC = REPO_ROOT / "packages" / "openpi-client" / "src"
CLIENT_INFERENCE_SRC = REPO_ROOT / "client" / "inference"

for path in (OPENPI_CLIENT_SRC, CLIENT_INFERENCE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mock_action_server import MockActionChunkServer  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402


def test_mock_action_server_returns_generated_action_chunk():
    server = MockActionChunkServer(host="127.0.0.1", port=0, action_horizon=4, action_dim=3)
    server.start_in_thread()
    try:
        policy = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=server.port)

        metadata = policy.get_server_metadata()
        first = policy.infer({"action_start": 10.0})
        second = policy.infer({"action_start": 10.0})

        assert metadata["server_type"] == "mock_action_chunk_server"
        np.testing.assert_allclose(
            first["actions"],
            np.asarray(
                [
                    [10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0],
                    [19.0, 20.0, 21.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            second["actions"],
            np.asarray(
                [
                    [22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0],
                    [28.0, 29.0, 30.0],
                    [31.0, 32.0, 33.0],
                ],
                dtype=np.float32,
            ),
        )
        assert first["server_timing"]["request_id"] == 1
        assert second["server_timing"]["request_id"] == 2
        assert "request_index" not in first["server_timing"]
        assert first["client_timing"]["request_id"] == 1
    finally:
        server.close()


def test_mock_action_server_can_echo_explicit_mock_actions():
    server = MockActionChunkServer(host="127.0.0.1", port=0, action_horizon=2, action_dim=2)
    server.start_in_thread()
    try:
        policy = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=server.port)

        response = policy.infer({"mock_actions": [[1.5, 2.5], [3.5, 4.5]]})

        np.testing.assert_allclose(response["actions"], np.asarray([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32))
    finally:
        server.close()
