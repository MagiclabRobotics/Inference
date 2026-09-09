"""模型推理与数据 transforms 的统一 Policy 封装。

Policy 位于网络服务和具体模型之间：网络层只传 dict，Policy 将其转换为模型
Observation、调用 JAX/PyTorch 采样函数，再把模型空间动作转换回机器人空间。
"""

from collections.abc import Sequence
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
from loguru import logger
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_rtc
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _supports_snapflow_inference(model: Any, *, is_pytorch: bool) -> bool:
    if not hasattr(model, "sample_actions_one_step"):
        return False
    if is_pytorch:
        return True
    return bool(getattr(model, "snapflow_enabled", False))


def _request_num_denoising_steps(obs: dict) -> int | None:
    for key in ("num_steps", "num_denoising_steps", "denoising_steps"):
        if key not in obs or obs[key] is None:
            continue
        try:
            value = int(obs[key])
        except Exception as exc:
            raise ValueError(f"{key} must be convertible to int, got {obs[key]!r}") from exc
        if value < 1:
            raise ValueError(f"{key} must be >= 1, got {obs[key]!r}")
        return value
    return None


class Policy(BasePolicy):
    """统一封装 JAX/PyTorch、标准采样、Legato 和 SnapFlow 的推理入口。"""

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._use_legato_inference = bool(self._sample_kwargs.pop("use_legato_inference", False))
        self._use_ttrtc_inference = bool(self._sample_kwargs.pop("use_ttrtc_inference", False))
        self._use_snapflow_inference = bool(self._sample_kwargs.pop("use_snapflow_inference", False))
        enabled_special_paths = sum(
            bool(value)
            for value in (
                self._use_legato_inference,
                self._use_ttrtc_inference,
                self._use_snapflow_inference,
            )
        )
        if enabled_special_paths > 1:
            raise ValueError(
                "Special inference paths cannot both be enabled; choose only one of "
                "use_legato_inference, use_ttrtc_inference, and use_snapflow_inference."
            )
        if self._use_ttrtc_inference and self._is_pytorch_model:
            raise ValueError("TTRTC inference is supported only by the JAX Pi0/Pi0.5 policy path.")
        self._bound_legato = False
        self._bound_ttrtc = False
        self._bound_snapflow = False
        self._bound_rtc = isinstance(model, pi0_rtc.Pi0RTC)
        if self._bound_rtc and (enabled_special_paths or self._is_pytorch_model):
            raise ValueError("RTC requires the JAX Pi0RTC path without Legato, TTRTC, or SnapFlow flags.")

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions_legato_raw = model.legato_sample_actions if hasattr(model, "legato_sample_actions") else None
            self._sample_actions_standard = model.sample_actions
            self._sample_actions_snapflow_raw = (
                model.sample_actions_one_step if hasattr(model, "sample_actions_one_step") else None
            )
            if self._use_snapflow_inference and _supports_snapflow_inference(model, is_pytorch=True):
                self._sample_actions = model.sample_actions_one_step
                self._bound_snapflow = True
            elif self._use_legato_inference and hasattr(model, "legato_sample_actions"):
                self._sample_actions = model.legato_sample_actions
                self._bound_legato = True
            else:
                if self._use_snapflow_inference:
                    logger.warning(
                        "use_snapflow_inference=True but model has no sample_actions_one_step; falling back."
                    )
                if self._use_legato_inference:
                    logger.warning("use_legato_inference=True but model has no legato_sample_actions; falling back.")
                self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions_legato_raw = (
                model.legato_sample_actions if hasattr(model, "legato_sample_actions") else None
            )
            self._sample_actions_ttrtc_raw = (
                model.ttrtc_sample_actions if hasattr(model, "ttrtc_sample_actions") else None
            )
            static_argnames = ("num_steps",)
            if self._bound_rtc:
                # RTC uses Python branches for booleans and schedule names inside the sampler.
                static_argnames += ("enable_rtc", "mask_prefix_delay", "prefix_attention_schedule")
            self._sample_actions_standard = nnx_utils.module_jit(
                model.sample_actions, static_argnames=static_argnames
            )
            self._sample_actions_snapflow_raw = (
                model.sample_actions_one_step if hasattr(model, "sample_actions_one_step") else None
            )
            if self._use_snapflow_inference and _supports_snapflow_inference(model, is_pytorch=False):
                self._sample_actions = nnx_utils.module_jit(model.sample_actions_one_step)
                self._bound_snapflow = True
            elif self._use_ttrtc_inference:
                if not hasattr(model, "ttrtc_sample_actions"):
                    raise ValueError(
                        "use_ttrtc_inference=True requires Pi0TTRTC with ttrtc_sample_actions; "
                        f"got {type(model).__name__}."
                    )
                self._sample_actions = nnx_utils.module_jit(
                    model.ttrtc_sample_actions,
                    static_argnames=("num_steps",),
                )
                self._bound_ttrtc = True
            elif self._use_legato_inference and hasattr(model, "legato_sample_actions"):
                self._sample_actions = nnx_utils.module_jit(
                    model.legato_sample_actions,
                    static_argnames=("num_steps",),
                )
                self._bound_legato = True
            else:
                if self._use_snapflow_inference:
                    logger.warning(
                        "use_snapflow_inference=True but model has no sample_actions_one_step; falling back."
                    )
                if self._use_legato_inference:
                    logger.warning("use_legato_inference=True but model has no legato_sample_actions; falling back.")
                self._sample_actions = self._sample_actions_standard
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        """把一份机器人观测转换为可执行的 action chunk。"""
        if bool(obs.get("enable_rtc", False)) and not self._bound_rtc:
            raise ValueError("RTC requires a Pi0RTC policy, e.g. policy.config=pi05_rtc_flatten_fold_inference.")
        if self._bound_rtc:
            self._validate_rtc_request(obs)
        # 1. 输入 transforms 完成字段重排、图像/语言处理、padding 和归一化。
        #    先浅复制树结构，避免 transform 就地修改网络层收到的原始 payload。
        logger.info("---")
        start_time = time.monotonic()
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if self._bound_rtc:
            # Sampling controls are kwargs, not Observation tensors (the schedule is a string).
            for key in (
                "prev_action_chunk_model", "enable_rtc", "mask_prefix_delay", "prefix_attention_schedule",
                "max_guidance_weight", "inference_delay", "execute_horizon",
            ):
                inputs.pop(key, None)
        logger.info("input transform: {}", 1000 * (time.monotonic() - start_time))
        step_time = time.monotonic()
        if not self._is_pytorch_model:
            # 2. 网络请求是一条样本；模型始终按 batch 形式运行，因此补 batch 维。
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # 3. 将客户端请求中的实时参数转成模型 sample_actions 的关键字参数。
        sample_kwargs = dict(self._sample_kwargs)
        if bool(obs.get("enable_ttrtc", False)) and not self._bound_ttrtc:
            raise ValueError(
                "Client requested TTRTC, but the policy server was not started with "
                "use_ttrtc_inference=True."
            )
        request_num_steps = _request_num_denoising_steps(obs)
        if request_num_steps is not None and not self._bound_snapflow:
            sample_kwargs["num_steps"] = request_num_steps

        if not self._bound_snapflow:
            # RTC/Legato 使用上一 chunk 和延迟估计引导本次动作生成。
            prev_action_chunk_model = obs.get("prev_action_chunk_model")
            if prev_action_chunk_model is not None:
                prev_actions = (
                    torch.from_numpy(np.asarray(prev_action_chunk_model)).to(self._pytorch_device)
                    if self._is_pytorch_model
                    else jnp.asarray(prev_action_chunk_model)
                )
                if prev_actions.ndim == 2:
                    prev_actions = prev_actions[None, ...]
                sample_kwargs[
                    "prev_actions" if (self._bound_legato or self._bound_ttrtc) else "prev_action_chunk"
                ] = prev_actions
            elif obs.get("prev_action_chunk") is not None:
                if self._bound_legato or self._bound_ttrtc:
                    sample_kwargs["prev_actions"] = obs["prev_action_chunk"]
                else:
                    sample_kwargs["prev_action_chunk"] = obs["prev_action_chunk"]
            if "inference_delay" in obs:
                sample_kwargs["inference_delay"] = obs["inference_delay"]
            if self._bound_legato and "ramp_down" in obs:
                sample_kwargs["ramp_down"] = obs["ramp_down"]
            if "execute_horizon" in obs:
                sample_kwargs["execute_horizon"] = obs["execute_horizon"]
        if self._bound_rtc:
            for key in ("enable_rtc", "mask_prefix_delay", "prefix_attention_schedule", "max_guidance_weight"):
                if key in obs:
                    sample_kwargs[key] = float(obs[key]) if key == "max_guidance_weight" else obs[key]
        if noise is not None:
            step_time = time.monotonic()
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise
            logger.info("noise process: {}", 1000 * (time.monotonic() - step_time))

        # 4. dict 到此变成模型定义的强结构 Observation。
        observation = _model.Observation.from_dict(inputs)
        logger.info("input postprocess: {}", 1000 * (time.monotonic() - step_time))
        step_time = time.monotonic()
        sample_actions_fn = self._sample_actions
        if self._bound_snapflow:
            snapflow_num_steps = request_num_steps or int(sample_kwargs.pop("num_steps", 1))
            if snapflow_num_steps > 1:
                sample_actions_fn = self._sample_actions_standard
                sample_kwargs["num_steps"] = snapflow_num_steps
            else:
                sample_kwargs.pop("num_steps", None)
        if self._bound_legato and sample_kwargs.get("prev_actions") is None:
            sample_actions_fn = self._sample_actions_standard
            for key in ("prev_actions", "inference_delay", "ramp_down", "execute_horizon"):
                sample_kwargs.pop(key, None)
        if self._bound_ttrtc and sample_kwargs.get("prev_actions") is None:
            # Cold start is standard denoising: an all-zero reference with a zero
            # prefix never enters the model as conditioning information.
            sample_kwargs["inference_delay"] = 0
            sample_kwargs["execute_horizon"] = 0
            sample_kwargs["prev_actions"] = jnp.zeros(
                (observation.state.shape[0], self._model.action_horizon, self._model.action_dim),
                dtype=observation.state.dtype,
            )
        # 5. 得到归一化模型空间中的 [batch, horizon, action_dim] 动作。
        actions_model = sample_actions_fn(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        model_time = time.monotonic() - step_time
        logger.info("sampling time: {}", model_time * 1000)
        step_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": actions_model,
        }
        if self._is_pytorch_model:
            actions_model_np = np.asarray(actions_model[0, ...].detach().cpu(), dtype=np.float32)
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            # JAX inference commonly returns bfloat16.  The websocket MsgPack
            # codec intentionally supports portable NumPy dtypes only, so
            # convert transport-facing actions to float32 here.
            actions_model_np = np.asarray(actions_model[0, ...], dtype=np.float32)
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # 6. 去掉 batch 维后执行反归一化和机器人输出裁剪（Agilex 最终取前 14 维）。
        outputs = self._output_transform(outputs)
        outputs["actions"] = np.asarray(outputs["actions"], dtype=np.float32)
        logger.info("output transform: {}", 1000 * (time.monotonic() - step_time))
        # Legato/TTRTC 需要上一轮的模型空间动作，所以除可执行 actions 外一并返回原始结果。
        outputs["actions_model"] = actions_model_np
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        logger.info("---- infer_ms: {}", 1000 * (time.monotonic() - start_time))
        return outputs

    def _validate_rtc_request(self, obs: dict) -> None:
        if obs.get("prev_action_chunk") is not None:
            raise ValueError(
                "RTC requires normalized prev_action_chunk_model; legacy prev_action_chunk is ambiguous. "
                "Update the RTC client to use actions_model returned by the server."
            )
        prev = obs.get("prev_action_chunk_model")
        if prev is not None:
            arr = np.asarray(prev)
            expected = (self._model.action_horizon, self._model.action_dim)
            if arr.shape not in (expected, (1, *expected)) or not np.isfinite(arr).all():
                raise ValueError(f"RTC prev_action_chunk_model must be finite with shape {expected} or {(1, *expected)}.")
        for key in ("enable_rtc", "mask_prefix_delay"):
            if key in obs and not isinstance(obs[key], (bool, np.bool_)):
                raise ValueError(f"RTC {key} must be a boolean.")
        if "prefix_attention_schedule" in obs and obs["prefix_attention_schedule"] not in {
            "ones", "zeros", "linear", "exp"
        }:
            raise ValueError("RTC prefix_attention_schedule must be ones, zeros, linear, or exp.")
        if "max_guidance_weight" in obs:
            weight = float(obs["max_guidance_weight"])
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("RTC max_guidance_weight must be finite and non-negative.")

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logger.info("Dumping policy records to: {}", record_dir)
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
