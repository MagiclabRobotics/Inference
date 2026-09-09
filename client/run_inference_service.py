#!/usr/bin/env python3
"""启动推理服务 TCP（对接 piperserver-master_xh use_inference_service）。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


CLIENT_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = CLIENT_DIR / "inference"
OPENPI_CLIENT_SRC = CLIENT_DIR.parent / "packages" / "openpi-client" / "src"

for path in (str(OPENPI_CLIENT_SRC), str(INFERENCE_DIR), str(CLIENT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        default=str(CLIENT_DIR / "config_agilex.yaml"),
        help="yaml 路径（collector/ROS/策略；inference.mode 可被 --mode 覆盖）",
    )
    pre_args, _ = pre.parse_known_args()

    from config import apply_mode_override, format_mode_list, load_config
    from integration.runtime_builder import resolve_config_path

    cfg_path = resolve_config_path(pre_args.config)
    cfg_preview = load_config(cfg_path)
    mode_choices = cfg_preview.available_modes()

    parser = argparse.ArgumentParser(
        description="Inference service for piperserver collector",
        parents=[pre],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_inference_service.py --mode temporal_smoothing\n"
            "  python run_inference_service.py --mode vlash_async --config config_agilex.yaml\n"
            "  python run_inference_service.py --list-modes\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=mode_choices,
        default=None,
        metavar="MODE",
        help=f"覆盖 yaml inference.mode；可选: {', '.join(mode_choices)}",
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="打印配置中的策略列表与说明后退出",
    )
    parser.add_argument("--host", default=None, help="覆盖 inference_service.listen_host")
    parser.add_argument("--port", type=int, default=None, help="覆盖 inference_service.listen_port")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    from config import section
    from integration.inference_service import InferenceService, InferenceServiceTcpServer

    cfg = load_config(resolve_config_path(args.config))
    if args.list_modes:
        if args.mode is not None:
            apply_mode_override(cfg, args.mode)
        print(format_mode_list(cfg))
        return

    log = logging.getLogger(__name__)
    # 仅在显式 --mode 时覆盖；否则保留 yaml 的 execution_mode/async_mode（master 模式体系）。
    if args.mode is not None:
        startup_mode = apply_mode_override(cfg, args.mode)
        log.info("inference mode overridden by CLI: %s", startup_mode)
    else:
        startup_mode = None
        log.info("inference mode from config: %s", cfg.mode)

    svc_cfg = section(cfg, "inference_service")
    listen_host = args.host or svc_cfg.get("listen_host", "0.0.0.0")
    listen_port = int(args.port or svc_cfg.get("listen_port", 9001))

    service = InferenceService(
        default_config_path=str(resolve_config_path(args.config)),
        startup_mode=startup_mode,
    )
    server = InferenceServiceTcpServer(listen_host, listen_port, service)
    server.serve_forever()


if __name__ == "__main__":
    main()
