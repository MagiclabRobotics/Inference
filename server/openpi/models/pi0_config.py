import dataclasses
from typing import TYPE_CHECKING

from flax import traverse_util
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0
    from openpi.models.pi0_rtc import Pi0RTC
    from openpi.models.pi0_ttrtc import Pi0TTRTC


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # If True, append Legato schedule omega to per-step noisy action features.
    # This widens `action_in_proj` input from action_dim to action_dim + 1.
    # Set to False for ablation with the original architecture.
    action_condition_on_omega: bool = False
    # If True, add SnapFlow target-time conditioning layers and enable 1-NFE sampling.
    enable_snapflow: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Paper-style RL Token. This is different from the prompt-side RLToken text
    # conditioning: it adds an internal learnable prefix token. Stage1 can train
    # the token/head with BC; stage2 can add actor/critic heads on top.
    paper_rl_token: bool = False
    paper_rl_actor_critic: bool = False
    paper_rl_latent_dim: int = 256
    paper_rl_mlp_hidden_dim: int = 512
    paper_rl_actor_delta_scale: float = 0.10
    paper_rl_reference_num_steps: int = 4
    paper_rl_inference_num_steps: int | None = None
    paper_rl_use_decoded_token: bool = True
    paper_rl_actor_bc_weight: float = 1.0
    paper_rl_delta_l2_weight: float = 0.01
    paper_rl_critic_weight: float = 0.10
    paper_rl_recon_weight: float = 0.01
    paper_rl_actor_pg_weight: float = 1.0
    paper_rl_critic_td_weight: float = 1.0
    paper_rl_logprob_std: float = 0.10
    paper_rl_discount: float = 0.99
    paper_rl_advantage_clip: float = 5.0

    # DSRL-style diffusion steering. This trains a small policy network that
    # predicts the initial denoising noise for a frozen pi0/pi0.5 policy.
    dsrl_steering: bool = False
    dsrl_hidden_dim: int = 512
    dsrl_noise_scale: float = 1.0
    dsrl_train_num_steps: int = 4
    dsrl_inference_num_steps: int | None = None
    dsrl_noise_l2_weight: float = 1e-4

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "Pi0":
        try:
            return super().load(params, remove_extra_params=remove_extra_params)
        except ValueError as exc:
            message = str(exc)
            if "target_time_mlp_1" not in message and "target_time_mlp_2" not in message:
                raise

            model_shape = nnx.eval_shape(self.create, jax.random.key(0))
            expected = nnx.state(model_shape).to_pure_dict()
            expected_flat = traverse_util.flatten_dict(expected)
            params_flat = traverse_util.flatten_dict(params)
            inserted = False
            for key, expected_value in expected_flat.items():
                if key in params_flat or not key or key[0] not in {"target_time_mlp_1", "target_time_mlp_2"}:
                    continue
                params_flat[key] = jnp.zeros(expected_value.shape, dtype=expected_value.dtype)
                inserted = True
            if not inserted:
                raise
            return super().load(traverse_util.unflatten_dict(params_flat), remove_extra_params=remove_extra_params)

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


@dataclasses.dataclass(frozen=True)
class Pi0RTCConfig(Pi0Config):
    """Config for Pi0RTC (real-time control) model. Uses same architecture as Pi0/Pi05 but sample_actions supports
    prev_action_chunk, inference_delay, execute_horizon for RTC guidance. Use this config when serving
    for RTC inference (e.g. agilex_inference_openpi_rtc.py). Set pi05=True for Pi05-based RTC (model_type PI05_RTC)."""

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05_RTC if self.pi05 else _model.ModelType.PI0_RTC

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0RTC":
        from openpi.models.pi0_rtc import Pi0RTC

        return Pi0RTC(self, rngs=nnx.Rngs(rng))

    @override
    def load_pytorch(self, train_config, weight_path: str):
        """RTC model is JAX-only; use a JAX checkpoint with serve_policy and Pi0RTCConfig."""
        raise NotImplementedError(
            "Pi0RTC is only supported with JAX checkpoints. Use a checkpoint saved from OpenPi JAX training "
            "(params directory, not model.safetensors) and serve with --policy.config=pi05_rtc_flatten_fold_inference (or your RTC config name)."
        )


@dataclasses.dataclass(frozen=True)
class Pi0TTRTCConfig(Pi0Config):
    """Inference-only Pi0/Pi0.5 config with TTRTC prefix sampling.

    ``model_type`` intentionally remains PI0/PI05 because the architecture and
    inference transforms are unchanged from the checkpoint that produced the
    weights.  This repository does not define a TTRTC training configuration.
    """

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0TTRTC":
        from openpi.models.pi0_ttrtc import Pi0TTRTC

        return Pi0TTRTC(self, rngs=nnx.Rngs(rng))

    @override
    def load_pytorch(self, train_config, weight_path: str):
        del train_config, weight_path
        raise NotImplementedError("TTRTC inference is supported only for JAX checkpoints.")


@dataclasses.dataclass(frozen=True)
class AdvantageEstimatorConfig(Pi0Config):
    # * Custom
    loss_action_weight: float = 1.0
    loss_value_weight: float = 1.0
