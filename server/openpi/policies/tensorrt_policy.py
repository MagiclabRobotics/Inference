from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import pathlib
import threading
import time
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
import torch

from openpi import transforms
from openpi.models import model as model_lib
from openpi.training import config as train_config_lib
from typing import Union, List
from openpi.shared import array_typing as at
import jax.numpy as jnp
def _import_tensorrt():
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT backend requested, but the 'tensorrt' Python package is not installed in this environment."
        ) from exc
    return trt


def _dtype_to_torch(trt: Any, dtype: Any) -> torch.dtype:
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if hasattr(trt, "bfloat16") and dtype == trt.bfloat16:
        return torch.bfloat16
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int64:
        return torch.int64
    if dtype == trt.bool:
        return torch.bool
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_uint8_chw(image: Any) -> np.ndarray:
    img = _to_numpy(image)
    if img.ndim == 4:
        img = img[-1]
    if img.ndim != 3:
        raise ValueError(f"Image must have ndim=3, got {img.shape}")
    if img.shape[-1] in (1, 3, 4) and img.shape[0] not in (1, 3, 4):
        img = np.transpose(img, (2, 0, 1))
    if np.issubdtype(img.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(img)) <= 1.5 else 1.0
        img = np.clip(img * scale, 0, 255).round().astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class TensorRTEngineRunner:
    def __init__(self, engine_path: pathlib.Path | str, *, device: str = "cuda", output_name: str = "actions") -> None:
        self.engine_path = pathlib.Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"TensorRT execution requires a CUDA device, got {device!r}")
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT execution requires torch.cuda.is_available()")

        self.trt = _import_tensorrt()
        with torch.cuda.device(self.device):
            self.logger = self.trt.Logger(self.trt.Logger.ERROR)
            runtime = self.trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
            if self.engine is None:
                raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Failed to create TensorRT execution context")

        self.tensor_names = tuple(self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors))
        self.input_names = tuple(
            name for name in self.tensor_names if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT
        )
        output_names = tuple(
            name for name in self.tensor_names if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.OUTPUT
        )
        if output_name not in output_names:
            if len(output_names) != 1:
                raise ValueError(f"Output tensor {output_name!r} not found. Available outputs: {output_names}")
            output_name = output_names[0]
        self.output_name = output_name
        self.lock = threading.Lock()
        self.io = self._describe_io()

    def _describe_io(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "mode": str(self.engine.get_tensor_mode(name)),
                "shape": list(self.engine.get_tensor_shape(name)),
                "dtype": str(self.engine.get_tensor_dtype(name)),
            }
            for name in self.tensor_names
        }

    def __call__(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tensors: dict[str, torch.Tensor] = {}
        with self.lock, torch.cuda.device(self.device):
            for name in self.input_names:
                if name not in inputs:
                    raise KeyError(f"TensorRT engine input {name!r} missing from prepared policy inputs")
                dtype = _dtype_to_torch(self.trt, self.engine.get_tensor_dtype(name))
                tensor = inputs[name].to(device=self.device, dtype=dtype).contiguous()
                if any(dim < 0 for dim in self.engine.get_tensor_shape(name)):
                    self.context.set_input_shape(name, tuple(tensor.shape))
                tensors[name] = tensor
                self.context.set_tensor_address(name, int(tensor.data_ptr()))

            out_shape = tuple(self.context.get_tensor_shape(self.output_name))
            if any(dim < 0 for dim in out_shape):
                out_shape = tuple(self.engine.get_tensor_shape(self.output_name))
            if any(dim < 0 for dim in out_shape):
                raise RuntimeError(f"Unresolved TensorRT output shape for {self.output_name}: {out_shape}")
            out_dtype = _dtype_to_torch(self.trt, self.engine.get_tensor_dtype(self.output_name))
            out = torch.empty(out_shape, dtype=out_dtype, device=self.device)
            self.context.set_tensor_address(self.output_name, int(out.data_ptr()))

            stream = torch.cuda.current_stream(self.device)
            ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned false")
            stream.synchronize()
            return out.to(torch.float32).detach()


class TensorRTPolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        *,
        engine_path: pathlib.Path | str,
        config_name: str,
        assets_dir: pathlib.Path | str | None = None,
        asset_id: str | None = None,
        default_prompt: str | None = None,
        device: str = "cuda",
        seed: int | None = None,
        output_name: str = "actions",
        precision: str = "fp16",
        use_legato_inference: bool = False,
    ) -> None:
        cfg = train_config_lib.get_config(config_name)
        if assets_dir is not None or asset_id is not None:
            assets = train_config_lib.AssetsConfig(
                assets_dir=None if assets_dir is None else str(assets_dir),
                asset_id=asset_id,
            )
            cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, assets=assets))

        data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
        if data_config.norm_stats is None:
            raise RuntimeError(
                "Norm stats are required for TensorRT deployment. "
                f"config={config_name!r}, assets_dir={assets_dir or cfg.assets_dirs!s}, "
                f"asset_id={data_config.asset_id!r}"
            )

        self._input_transform = transforms.compose(
            [
                transforms.InjectDefaultPrompt(default_prompt),
                *data_config.data_transforms.inputs,
                transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )
        self._output_transform = transforms.compose(
            [
                *data_config.model_transforms.outputs,
                transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ]
        )
        self._runner = TensorRTEngineRunner(engine_path, device=device, output_name=output_name)
        self._rng = np.random.default_rng(seed)
        self._action_horizon = int(cfg.model.action_horizon)
        self._action_dim = int(cfg.model.action_dim)
        self._metadata = {
            **(cfg.policy_metadata or {}),
            "backend": "tensorrt",
            "precision": precision,
            "engine": str(engine_path),
            "config": config_name,
            "asset_id": data_config.asset_id,
            "action_horizon": self._action_horizon,
            "action_dim": self._action_dim,
            "fixed_shapes": self._runner.io,
        }
        self.use_legato_inference = use_legato_inference

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _prepare_observation(self, raw_obs: dict[str, Any]) -> model_lib.Observation[torch.Tensor]:
        obs = dict(raw_obs)
        if "images" in obs:
            obs["images"] = {key: _to_uint8_chw(value) for key, value in obs["images"].items()}
        obs["state"] = _to_numpy(obs["state"]).reshape(-1).astype(np.float32)
        transformed = self._input_transform(obs)
        torch_inputs: dict[str, Any] = {}
        for key, value in transformed.items():
            if isinstance(value, dict):
                torch_inputs[key] = {
                    item_key: torch.from_numpy(np.asarray(item_value).copy())[None, ...]
                    for item_key, item_value in value.items()
                }
            else:
                torch_inputs[key] = torch.from_numpy(np.asarray(value).copy())[None, ...]
        return model_lib.Observation.from_dict(torch_inputs)

    def _prepare_noise(self, raw_obs: dict[str, Any]) -> torch.Tensor:
        if "noise" in raw_obs:
            noise = _to_numpy(raw_obs["noise"]).astype(np.float32)
            if noise.ndim == 2:
                noise = noise[None, ...]
        else:
            noise = self._rng.standard_normal((1, self._action_horizon, self._action_dim), dtype=np.float32)
        expected = (1, self._action_horizon, self._action_dim)
        if noise.shape != expected:
            raise ValueError(f"noise must have shape {expected[1:]} or {expected}, got {noise.shape}")
        return torch.from_numpy(noise)

    def _flatten_engine_inputs(
        self,
        observation: model_lib.Observation[torch.Tensor],
        raw_obs: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        inputs: dict[str, torch.Tensor] = {
            "state": observation.state.to(torch.float32).contiguous().cuda(),
            "noise": self._prepare_noise(raw_obs).to(torch.float32).contiguous().cuda(),
        }
        optional = {
            "lang_tokens": observation.tokenized_prompt,
            "lang_masks": observation.tokenized_prompt_mask,
            "tokenized_prompt_state_owner": observation.tokenized_prompt_state_owner,
            "token_ar_mask": observation.token_ar_mask,
            "token_loss_mask": observation.token_loss_mask,
        }
        for key, value in optional.items():
            if value is not None:
                inputs[key] = value.contiguous().cuda()
        __images_list = []
        __images_keys = []
        for key, value in observation.images.items():
            # inputs[f"image__{key}"] = value.to(torch.float32)
            __images_list.append(value.to(torch.float32).contiguous().cuda())
        for key, value in observation.image_masks.items():
            # inputs[f"image_mask__{key}"] = value.to(torch.bool)
            __images_keys.append(value.to(torch.bool).contiguous().cuda())
        inputs["images"] = torch.cat(__images_list, dim=1).contiguous()
        inputs["img_masks"] = torch.cat(__images_keys, dim=0).contiguous()
        return inputs

    def build_legato_schedule_jax(
        self,
        horizon: int,
        inference_delay: at.Int[at.Array, "*b"] | int,
        ramp_down: at.Int[at.Array, "*b"] | int,
    ) -> at.Float[at.Array, "*b h"]:
        """Build Legato continuation schedule omega in [0, 1] with shape [..., H]."""
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        delay = jnp.clip(jnp.asarray(inference_delay, dtype=jnp.int32), 0, horizon)
        max_ramp = jnp.maximum(horizon - delay, 0)
        ramp = jnp.clip(jnp.asarray(ramp_down, dtype=jnp.int32), 0, max_ramp)

        pos = jnp.arange(horizon, dtype=jnp.int32)
        full_mask = pos < delay[..., None]
        ramp_mask = jnp.logical_and(pos >= delay[..., None], pos < (delay + ramp)[..., None])
        ramp_denom = jnp.maximum(ramp[..., None], 1)
        ramp_progress = (pos - delay[..., None] + 1).astype(jnp.float32) / ramp_denom.astype(jnp.float32)
        ramp_values = jnp.clip(1.0 - ramp_progress, 0.0, 1.0)

        omega = jnp.where(full_mask, 1.0, 0.0)
        omega = jnp.where(ramp_mask, ramp_values, omega)
        return omega.astype(jnp.float32)

    def __perpare_legato_inputs(self, obs: dict[str, Any], inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prev_action_chunk_model = obs.get("prev_action_chunk_model")
        prev_actions = None
        if prev_action_chunk_model is not None:
            prev_actions = (jnp.asarray(prev_action_chunk_model))
            if prev_actions.ndim == 2:
                prev_actions = prev_actions[None, ...]
        elif obs.get("prev_action_chunk") is not None:
            prev_actions = obs["prev_action_chunk"]
        if prev_actions is None:
            # print(f"Using default prev_actions with shape {(1, self._action_horizon, self._action_dim)}")
            inputs["a_ref"] = torch.zeros((1, self._action_horizon, self._action_dim), dtype=torch.float32).contiguous().cuda()
            inputs["omega"] = torch.zeros((1, self._action_horizon, 1), dtype=torch.float32).contiguous().cuda()
            return inputs
        execute_horizon = obs.get("execute_horizon")
        # print(f"execute_horizon: {execute_horizon}, prev_actions shape: {prev_actions.shape if prev_actions is not None else None}")
        if prev_actions is not None:
            s = jnp.asarray(execute_horizon, dtype=jnp.int32)
            s = jnp.clip(s, 1, self._action_horizon)
            pos = jnp.arange(self._action_horizon, dtype=jnp.int32)
            if s.ndim == 0:
                gather_idx = jnp.clip(pos + s, 0, self._action_horizon - 1)
                a_ref = jnp.take(prev_actions, gather_idx, axis=1)
            else:
                s = jnp.broadcast_to(s.reshape(-1), (1,))
                gather_idx = jnp.clip(pos[None, :] + s[:, None], 0, self._action_horizon - 1)
                a_ref = jnp.take_along_axis(prev_actions, gather_idx[..., None], axis=1)
            a_ref_np = np.asarray(a_ref, dtype=np.float32)
            inputs["a_ref"] = torch.from_numpy(a_ref_np.copy()).contiguous().cuda()

        inference_delay = obs.get("inference_delay")
        ramp_down = obs.get("ramp_down")
        # print(f"inference_delay: {inference_delay}, ramp_down: {ramp_down}")
        omega = self.build_legato_schedule_jax(
            horizon=self._action_horizon,
            inference_delay=inference_delay,
            ramp_down=ramp_down
        )
        omega = jnp.broadcast_to(omega, (1, self._action_horizon))
        omega_expanded = omega[..., None]
        # print(f"omega shape: {omega.shape}, omega_expanded shape: {omega_expanded.shape}")
        omega_expanded_np = np.asarray(omega_expanded, dtype=np.float32)
        inputs["omega"] = torch.from_numpy(omega_expanded_np.copy()).contiguous().cuda()
        return inputs

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        start = time.monotonic()
        observation = self._prepare_observation(obs)
        inputs = self._flatten_engine_inputs(observation, obs)
        if self.use_legato_inference:
            inputs = self.__perpare_legato_inputs(obs, inputs)
        infer_start = time.monotonic()
        actions_model = self._runner(inputs).cpu().numpy()
        engine_ms = (time.monotonic() - infer_start) * 1000.0

        outputs = {
            "state": observation.state.cpu().numpy()[0],
            "actions": actions_model[0],
        }
        outputs = self._output_transform(outputs)
        outputs["actions_model"] = actions_model[0]
        outputs["policy_timing"] = {
            "engine_ms": engine_ms,
            "infer_ms": (time.monotonic() - start) * 1000.0,
        }
        return outputs
