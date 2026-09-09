import dataclasses
import pathlib

import jax
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


REAL_DATASET_REPO = pathlib.Path(__file__).parent / "assets" / "task_a_base_episode001_subset"


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
        framework="pytorch",
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, framework="pytorch")
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
        num_workers=2,
        framework="pytorch",
    )
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.TrainConfig(
        name="fake_test",
        model=pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48),
        batch_size=4,
        num_workers=0,
    )

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2, framework="pytorch")
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    if not REAL_DATASET_REPO.exists():
        pytest.skip(f"Real dataset not found: {REAL_DATASET_REPO}")

    config = _config.get_config("pi05_flatten_fold_normal_follow_paper")
    config = dataclasses.replace(
        config,
        data=_config.LerobotAgilexDataConfig(
            repo_id=str(REAL_DATASET_REPO),
            default_prompt="Flatten and fold the cloth.",
            use_delta_joint_actions=False,
        ),
        batch_size=4,
        num_workers=0,
    )

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
        framework="pytorch",
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
