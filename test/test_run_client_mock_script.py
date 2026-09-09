from __future__ import annotations

from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_client_mock_dry_run_prints_mock_server_and_client_commands():
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "run_client_mock.sh"),
            "--dry-run",
            "--port",
            "19001",
            "--max-publish-step",
            "3",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "mock_action_server.py" in result.stdout
    assert "scripts/run_client.sh --config" in result.stdout
    assert "port: 19001" in result.stdout
    assert "max_publish_step: 3" in result.stdout
