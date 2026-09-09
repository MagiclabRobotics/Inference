from __future__ import annotations

from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_server_dry_run_includes_snapflow_from_yaml(tmp_path):
    config = tmp_path / "server.yaml"
    config.write_text(
        """
python_bin: python3
port: 18000
use_snapflow_inference: true
policy:
  type: default
""".strip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_server.sh"), "--config", str(config), "--dry-run"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--use-snapflow-inference" in result.stdout


def test_run_server_dry_run_includes_ttrtc_from_yaml(tmp_path):
    config = tmp_path / "server.yaml"
    config.write_text(
        """
python_bin: python3
port: 18000
use_ttrtc_inference: true
policy:
  type: default
""".strip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_server.sh"), "--config", str(config), "--dry-run"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--use-ttrtc-inference" in result.stdout


def test_run_server_dry_run_includes_snapflow_from_json_override(tmp_path):
    config = tmp_path / "server.yaml"
    config.write_text(
        """
python_bin: python3
port: 18000
use_snapflow_inference: false
policy:
  type: default
""".strip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "run_server.sh"),
            "--config",
            str(config),
            "--json-config",
            '{"use_snapflow_inference": true}',
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--use-snapflow-inference" in result.stdout


def test_run_server_dry_run_includes_checkpoint_asset_id(tmp_path):
    config = tmp_path / "server.yaml"
    config.write_text(
        """
python_bin: python3
port: 18000
policy:
  type: checkpoint
  config: pi05_flatten_fold_normal_follow_paper
  dir: /tmp/checkpoint
  asset_id: magiclab
""".strip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_server.sh"), "--config", str(config), "--dry-run"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--policy.asset-id magiclab" in result.stdout
