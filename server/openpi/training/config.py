"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.agilex_policy as agilex_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from typing_extensions import override
import tyro

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)
    # Optional subset of episodes to load from a LeRobot dataset.
    episodes: list[int] | None = None

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI0_RTC:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05 | _model.ModelType.PI05_RTC:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type not in (ModelType.PI0, ModelType.PI0_RTC),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LerobotAgilexDataConfig(DataConfigFactory):
    """
    Configuration for the Agilex robot dataset.
    This config handles the data transforms for the Agilex robot's multi-camera setup and state/action space.
    """

    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    use_delta_joint_actions: bool = True

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    episodes: list[int] | None = None

    # Repack transforms to match the dataset keys to the expected format
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )

    # Action keys that will be used to read the action sequence from the dataset
    action_sequence_keys: Sequence[str] = ("action",)

    # mask state out (set to all zeros)
    mask_state: bool = False

    # if insert progress into prompt
    insert_advantage_into_prompt: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:

        # Use local variables instead of modifying frozen self
        default_prompt = self.default_prompt
        repack_transforms = self.repack_transforms

        # if prompt_from_task is True, set default_prompt to None and add prompt to data transforms
        if self.base_config and self.base_config.prompt_from_task:
            default_prompt = None
            original_repack = self.repack_transforms.inputs[0]
            new_structure = dict(original_repack.structure)
            new_structure["prompt"] = "prompt"
            repack_transforms = _transforms.Group(
                inputs=[_transforms.RepackTransform(new_structure)]
            )

        # Create data transforms for inputs and outputs
        data_transforms = _transforms.Group(
            inputs=[
                agilex_policy.AgilexInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    mask_state=self.mask_state,
                )
            ],
            outputs=[agilex_policy.AgilexOutputs()],
        )
        if self.insert_advantage_into_prompt:
            data_transforms.inputs.insert(0, _transforms.InsertAdvantageIntoPrompt())

        # Apply delta action transform if enabled
        if self.use_delta_joint_actions:
            # Assuming first 13 dimensions are joints and last dimension is gripper
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)  # index 6, 13 is gripper
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Create model transforms
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            episodes=self.episodes,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # VLASH shared-observation training switches.
    # When enabled, data loader returns [B, O, ...] action/state tensors and an offset mask.
    vlash_shared_observation: bool = False
    vlash_max_delay_steps: int = 0
    # Enable Legato loss (paper-aligned schedule randomization) for shared-observation training.
    use_legato_loss: bool = False
    # Legato schedule/training hyperparameters (Appendix Table A.3 defaults).
    legato_num_denoising_steps: int = 5
    legato_delay_range: tuple[int, int] = (0, 10)
    legato_ramp_range: tuple[int, int] = (0, 50)

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    
#************************advantage estimator***************************
    advantage_estimator: bool = False
    is_train: bool = True  # * Only use partial data in training
    # split:    str  = None  # one of ['train_tasks', 'val_tasks', 'heldout_tasks']
    # * Bugfix, only use train_tasks for training
    split: str = 'all'  # * Only use training tasks for training, choose from ['train', 'val', 'all']
    drop_last: bool = True  # If true, will drop the last incomplete batch.
    skip_norm_stats: bool = False
#************************advantage estimator***************************
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    # ----------------------- Normal π₀.5 full fine-tuning (see README § Preparation) -----------------------
    # Set repo_id to absolute path to ./data/<Task>/base and weight_loader to π₀.5 base checkpoint.
    # Then: compute_norm_states_fast.py --config-name <name>; train.py <name> --exp_name=<xxx>
        TrainConfig(
        name="pi05_flatten_fold_normal_follow_paper",
        model=pi0_config.Pi0Config(pi05=True),
        data = LerobotAgilexDataConfig(
            repo_id="OpenDriveLab-org/Kai0",
            assets=AssetsConfig(
                assets_dir="/path/to/assets/pi05_flatten_fold_normal_follow_paper",
                asset_id="OpenDriveLab-org/Kai0",
            ),
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=10_000,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        num_train_steps=80_000,
        keep_period=5000,
        num_workers=32,
        batch_size=128,
    ),
    TrainConfig(
        name="pi05_flatten_fold_normal",
        model=pi0_config.Pi0Config(pi05=True),
        data = LerobotAgilexDataConfig(
            repo_id="/path/to/data/Task_A/base",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("<path/to/pi05_base/checkpoint>"),
        num_train_steps=100_000,
        keep_period=5000,
        num_workers=8,
        batch_size=256,
    ),
    TrainConfig(
        name="pi05_flatten_fold_normal_vlash",
        model=pi0_config.Pi0Config(pi05=True, max_token_len=500),
        data = LerobotAgilexDataConfig(
            repo_id="/path/to/data/Task_A/base",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=10_000,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        num_train_steps=80_000,
        keep_period=5000,
        num_workers=32,
        batch_size=64,

        #VLASH
        vlash_shared_observation=True,
        vlash_max_delay_steps=4,
    ),
    TrainConfig(
        name="pi05_flatten_fold_normal_vlash_inference",
        model=pi0_config.Pi0Config(pi05=True, max_token_len=900),
        data = LerobotAgilexDataConfig(
            repo_id="magiclab",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/pfs/user/Models/openpi/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=10_000,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        num_train_steps=80_000,
        keep_period=5000,
        num_workers=4,
        batch_size=2,

        #VLASH
        vlash_shared_observation=True,
        vlash_max_delay_steps=8,

        #lora
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_flatten_fold_snapflow_jax_mixed1_inference",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="Task_A_base_dagger_202604302308_filter_speedup",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        batch_size=1,
        num_workers=1,
    ),
    TrainConfig(
        name="pi05_flatten_fold_normal_legato",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=60, action_condition_on_omega=True),
        data = LerobotAgilexDataConfig(
            repo_id="/pfs/user/data/Kai0/Task_A/base",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/pfs/user/Models/openpi/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=10_000,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        num_train_steps=80_000,
        keep_period=5000,
        num_workers=128,
        batch_size=128,

        # Legato
        use_legato_loss=True,
    ),

    TrainConfig(
        name="pi05_flatten_fold_normal_legato_inference",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=60, action_condition_on_omega=True),
        data = LerobotAgilexDataConfig(
            repo_id="magiclab",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/pfs/user/Models/openpi/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=10_000,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        num_train_steps=80_000,
        keep_period=5000,
        num_workers=128,
        batch_size=128,

        # Legato
        use_legato_loss=True,
    ),

    TrainConfig(
        name="pi05_flatten_fold_normal_legato_lr",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=60, action_condition_on_omega=True),
        data = LerobotAgilexDataConfig(
            repo_id="/pfs/user/data/Kai0/Task_A/base",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/pfs/user/Models/openpi/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr=5e-5,
            decay_lr=5e-6,
            decay_steps=30_000,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        num_train_steps=80_000,
        keep_period=5000,
        num_workers=128,
        batch_size=128,

        # Legato
        use_legato_loss=True,
    ),

    TrainConfig(
        name="pi05_flatten_fold_normal_legato_debug",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=60, action_condition_on_omega=True),
        data = LerobotAgilexDataConfig(
            repo_id="/pfs/user/data/Kai0/Task_A/base",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/pfs/user/Models/openpi/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=10_000,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        num_train_steps=80_000,
        keep_period=5000,
        num_workers=32,
        batch_size=64,
        
        # lora finetuning with legato loss for debugging
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,

        # Legato
        use_legato_loss=True,
    ),
    #**************************FlattenFold RTC Inference*******************************
    # Use this config when serving the policy for agilex_inference_openpi_rtc.py (JAX checkpoints only).
    TrainConfig(
        name="pi05_rtc_flatten_fold_inference",
        model=pi0_config.Pi0RTCConfig(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="OpenDriveLab-org/Kai0",
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("<path_to/pi05_base/checkpoint>"),
        num_train_steps=100_000,
        keep_period=5000,
        num_workers=8,
        batch_size=256,
    ),


    TrainConfig(
        name="pi05_fold_box_normal",
        model=pi0_config.Pi0Config(pi05=True),
        data = LerobotAgilexDataConfig(
            repo_id="Magiclab/Kai0",
            assets=AssetsConfig(
                assets_dir="/path/to/assets/pi05_flatten_fold_normal_follow_paper",
                asset_id="Magiclab/Kai0",
            ),
            default_prompt="Fold the paper box.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=10_000,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        num_train_steps=80_000,
        keep_period=5000,
        num_workers=32,
        batch_size=128,
    ),

    # **************************FoldBox RL/DSRL Inference*******************************
    TrainConfig(
        name="pi05_fold_box_rl_token_inference",
        model=pi0_config.Pi0Config(pi05=True),
        data=LerobotAgilexDataConfig(
            repo_id="Magiclab/Kai0",
            assets=AssetsConfig(
                assets_dir="assets/pi05_fold_box_0428",
                asset_id="Magiclab/Kai0",
            ),
            default_prompt="Fold the paper box., RLToken: very_positive, RLScore: +1.0000",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.NoOpWeightLoader(),
    ),
    TrainConfig(
        name="pi05_fold_box_dsrl_inference",
        model=pi0_config.Pi0Config(
            pi05=True,
            dsrl_steering=True,
            dsrl_hidden_dim=512,
            dsrl_noise_scale=1.0,
            dsrl_train_num_steps=4,
            dsrl_inference_num_steps=4,
        ),
        data=LerobotAgilexDataConfig(
            repo_id="Magiclab/Kai0",
            assets=AssetsConfig(
                assets_dir="assets/pi05_fold_box_0428",
                asset_id="Magiclab/Kai0",
            ),
            default_prompt="Fold the paper box.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.NoOpWeightLoader(),
    ),
    TrainConfig(
        name="pi05_fold_box_paper_rl_token_stage1_inference",
        model=pi0_config.Pi0Config(
            pi05=True,
            paper_rl_token=True,
            paper_rl_actor_critic=False,
            paper_rl_latent_dim=256,
            paper_rl_use_decoded_token=True,
            paper_rl_inference_num_steps=4,
        ),
        data=LerobotAgilexDataConfig(
            repo_id="Magiclab/Kai0",
            assets=AssetsConfig(
                assets_dir="assets/pi05_fold_box_0428",
                asset_id="Magiclab/Kai0",
            ),
            default_prompt="Fold the paper box.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.NoOpWeightLoader(),
    ),
    TrainConfig(
        name="pi05_fold_box_rl_token_actor_critic_direct_stage_inference",
        model=pi0_config.Pi0Config(
            pi05=True,
            paper_rl_token=True,
            paper_rl_actor_critic=True,
            paper_rl_latent_dim=256,
            paper_rl_mlp_hidden_dim=384,
            paper_rl_actor_delta_scale=0.10,
            paper_rl_reference_num_steps=2,
            paper_rl_inference_num_steps=4,
            paper_rl_use_decoded_token=True,
            paper_rl_actor_pg_weight=1.0,
            paper_rl_critic_td_weight=1.0,
            paper_rl_delta_l2_weight=0.01,
            paper_rl_logprob_std=0.10,
            paper_rl_discount=0.99,
        ),
        data=LerobotAgilexDataConfig(
            repo_id="Magiclab/Kai0",
            assets=AssetsConfig(
                assets_dir="assets/pi05_fold_box_dsrl",
                asset_id="Magiclab/Kai0",
            ),
            default_prompt="Fold the paper box.",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.NoOpWeightLoader(),
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
