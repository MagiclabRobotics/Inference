"""从训练配置和 checkpoint 组装可部署的 Policy。

训练配置不仅描述网络结构，也定义机器人数据如何映射到模型空间。部署时必须复用
同一套 transforms 和 norm stats，才能保证推理输入与训练分布一致。
"""

import logging
import dataclasses
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
from openpi.models import pi0_config
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # checkpoint 布局决定加载哪套后端：model.safetensors 为 PyTorch，
    # params/ 目录为 JAX/NNX。
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)
    use_ttrtc = bool(sample_kwargs and sample_kwargs.get("use_ttrtc_inference"))
    if use_ttrtc:
        if is_pytorch:
            raise ValueError("TTRTC inference requires a JAX checkpoint; PyTorch safetensors are not supported.")
        model_config = train_config.model
        if (
            not isinstance(model_config, pi0_config.Pi0TTRTCConfig)
            and type(model_config) is not pi0_config.Pi0Config
        ):
            raise ValueError(
                "TTRTC inference supports only the standard Pi0/Pi0.5 JAX architecture; "
                f"got {type(model_config).__name__}."
            )
        incompatible_features = [
            name
            for name in (
                "action_condition_on_omega",
                "enable_snapflow",
                "paper_rl_token",
                "paper_rl_actor_critic",
                "dsrl_steering",
            )
            if bool(getattr(model_config, name, False))
        ]
        if incompatible_features:
            raise ValueError(
                "TTRTC inference cannot be combined with model features: "
                + ", ".join(incompatible_features)
            )
        if type(model_config) is pi0_config.Pi0Config:
            train_config = dataclasses.replace(
                train_config,
                model=pi0_config.Pi0TTRTCConfig(**dataclasses.asdict(model_config)),
            )
    if not is_pytorch and sample_kwargs and sample_kwargs.get("use_snapflow_inference"):
        model_config = train_config.model
        if hasattr(model_config, "enable_snapflow"):
            train_config = dataclasses.replace(
                train_config,
                model=dataclasses.replace(model_config, enable_snapflow=True),
            )

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    # DataConfig 构造机器人字段重排、图像/语言 token 化等 transforms。
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    # 输入与输出流水线方向相反：
    # robot data -> repack/data transform -> normalize -> model transform -> model
    # model action -> model transform -> unnormalize -> data/repack transform -> robot action
    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
