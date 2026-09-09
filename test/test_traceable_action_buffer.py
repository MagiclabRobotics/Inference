from __future__ import annotations

import inspect
from pathlib import Path
import sys
import time

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIENT_INFERENCE_SRC = REPO_ROOT / "client" / "inference"

if str(CLIENT_INFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(CLIENT_INFERENCE_SRC))

from traceable_action_buffer import ActionValueSourceItem  # noqa: E402
from traceable_action_buffer import StreamActionBuffer  # noqa: E402
from traceable_action_buffer import TraceableAction  # noqa: E402
from traceable_action_buffer import TraceableActionChunk  # noqa: E402
from action_buffers import NaiveAsyncBuffer  # noqa: E402
from action_buffers import TemporalEnsemblingBuffer  # noqa: E402
from action_buffers import create_action_buffer  # noqa: E402


def src(chunk_id: int, step_index: int, value: float, weight: float = 1.0) -> ActionValueSourceItem:
    return ActionValueSourceItem(chunk_id, step_index, value, weight)


def test_action_value_source_items_are_namedtuples():
    chunk = TraceableActionChunk.from_array([[1.0, 2.0]], chunk_id=10)
    source_item = chunk[0].value_source[0][0]

    assert isinstance(source_item, ActionValueSourceItem)
    assert source_item.chunk_id == 10
    assert source_item.step_index == 0
    assert source_item.value == 1.0
    assert source_item.weight == 1.0
    assert source_item == (10, 0, 1.0, 1.0)


def test_traceable_action_chunk_tracks_mean_value_sources():
    b = TraceableActionChunk.from_array([[1.0, 3.0], [5.0, 7.0]], chunk_id=10)
    c = TraceableActionChunk.from_array([[9.0, 11.0], [13.0, 15.0]], chunk_id=20)

    a = TraceableActionChunk.mean(b, c, chunk_id=30)

    assert a.id == 30
    np.testing.assert_allclose(a.as_array(), [[5.0, 7.0], [9.0, 11.0]])
    assert a.value_sources() == [
        (
            (src(10, 0, 1.0, 0.5), src(20, 0, 9.0, 0.5)),
            (src(10, 0, 3.0, 0.5), src(20, 0, 11.0, 0.5)),
        ),
        (
            (src(10, 1, 5.0, 0.5), src(20, 1, 13.0, 0.5)),
            (src(10, 1, 7.0, 0.5), src(20, 1, 15.0, 0.5)),
        ),
    ]


def test_traceable_action_chunk_weighted_sum_accepts_per_step_weights():
    b = TraceableActionChunk.from_array([[0.0], [10.0], [20.0]], chunk_id=10)
    c = TraceableActionChunk.from_array([[100.0], [110.0], [120.0]], chunk_id=20)

    a = TraceableActionChunk.weighted_sum(
        ((np.array([1.0, 0.5, 0.0]), b), (np.array([0.0, 0.5, 1.0]), c)),
        chunk_id=30,
    )

    np.testing.assert_allclose(a.as_array(), [[0.0], [60.0], [120.0]])
    assert a.value_sources() == [
        ((src(10, 0, 0.0, 1.0), src(20, 0, 100.0, 0.0)),),
        ((src(10, 1, 10.0, 0.5), src(20, 1, 110.0, 0.5)),),
        ((src(10, 2, 20.0, 0.0), src(20, 2, 120.0, 1.0)),),
    ]


def test_traceable_action_chunk_tracks_slice_add_assignment_value_sources():
    b = TraceableActionChunk.from_array(np.arange(12, dtype=float).reshape(6, 2), chunk_id=10)
    c = TraceableActionChunk.from_array(np.arange(100, 112, dtype=float).reshape(6, 2), chunk_id=20)
    a = TraceableActionChunk.zeros((6, 2), chunk_id=30)

    a[1:5] = b[2:6] + c[0:4]

    np.testing.assert_allclose(a[1:5].as_array(), b[2:6].as_array() + c[0:4].as_array())
    assert a[1:5].value_sources() == [
        ((src(10, 2, 4.0), src(20, 0, 100.0)), (src(10, 2, 5.0), src(20, 0, 101.0))),
        ((src(10, 3, 6.0), src(20, 1, 102.0)), (src(10, 3, 7.0), src(20, 1, 103.0))),
        ((src(10, 4, 8.0), src(20, 2, 104.0)), (src(10, 4, 9.0), src(20, 2, 105.0))),
        ((src(10, 5, 10.0), src(20, 3, 106.0)), (src(10, 5, 11.0), src(20, 3, 107.0))),
    ]


def test_traceable_action_chunk_full_slice_returns_independent_chunk():
    chunk = TraceableActionChunk.from_array([[1.0, 2.0], [3.0, 4.0]], chunk_id=10)

    sliced = chunk[:]
    sliced[0] = [9.0, 9.0]

    assert isinstance(sliced, TraceableActionChunk)
    assert sliced.id == chunk.id
    np.testing.assert_allclose(sliced.as_array(), [[9.0, 9.0], [3.0, 4.0]])
    np.testing.assert_allclose(chunk.as_array(), [[1.0, 2.0], [3.0, 4.0]])


def test_stream_action_buffer_integration_keeps_chunk_level_names():
    source = inspect.getsource(StreamActionBuffer.integrate_new_chunk)

    assert "new_list" not in source
    assert "old_list" not in source


def test_traceable_action_chunk_popleft_removes_and_returns_first_action():
    chunk = TraceableActionChunk.from_array([[1.0, 2.0], [3.0, 4.0]], chunk_id=10)

    first = chunk.popleft()

    np.testing.assert_allclose(first.value, [1.0, 2.0])
    assert first.value_source == ((src(10, 0, 1.0),), (src(10, 0, 2.0),))
    np.testing.assert_allclose(chunk.as_array(), [[3.0, 4.0]])
    assert chunk.value_sources() == [((src(10, 1, 3.0),), (src(10, 1, 4.0),))]
    assert chunk.popleft().value.tolist() == [3.0, 4.0]
    with pytest.raises(IndexError):
        chunk.popleft()


def test_traceable_action_chunk_formats_source_map_for_terminal(capsys):
    b = TraceableActionChunk.from_array(np.zeros((5, 1)), chunk_id=1)
    c = TraceableActionChunk.from_array(np.ones((5, 1)), chunk_id=2)
    blended_prefix = TraceableActionChunk.weighted_sum(((0.5, b[0:3]), (0.5, c[0:3])), chunk_id=2)
    current = TraceableActionChunk([*blended_prefix, *c[3:5]], chunk_id=2)

    assert current.format_source_map(dim=0, cell_width=3) == "\n".join(
        [
            " 1  1  1       ",
            " 2  2  2  2  2 ",
            " □  □  □  □  □ ",
        ]
    )

    current.print_source_map(dim=0, cell_width=3)
    captured = capsys.readouterr()
    assert captured.out == " 1  1  1       \n 2  2  2  2  2 \n □  □  □  □  □ \n"


def test_traceable_action_chunk_source_map_prints_chunk_id_mod_10():
    b = TraceableActionChunk.from_array(np.zeros((5, 1)), chunk_id=10)
    c = TraceableActionChunk.from_array(np.ones((5, 1)), chunk_id=11)
    blended_prefix = TraceableActionChunk.weighted_sum(((0.5, b[0:3]), (0.5, c[0:3])), chunk_id=11)
    current = TraceableActionChunk([*blended_prefix, *c[3:5]], chunk_id=11)

    assert current.format_source_map(dim=0, cell_width=3) == "\n".join(
        [
            " 0  0  0       ",
            " 1  1  1  1  1 ",
            " □  □  □  □  □ ",
        ]
    )


def test_temporal_ensembling_blends_old_and_new_chunks():
    buffer = StreamActionBuffer(smooth_method="temporal_ensembling", ensemble_new_weight=0.75)
    assert buffer.integrate_new_chunk([[0.0], [0.0]], max_k=0, min_m=2) is not None
    assert buffer.integrate_new_chunk([[4.0], [4.0]], max_k=0, min_m=2) is not None

    first = buffer.pop_next_action()
    second = buffer.pop_next_action()

    assert first is not None
    assert second is not None
    assert isinstance(first["action"], TraceableAction)
    assert isinstance(second["action"], TraceableAction)
    assert first["action"].value == pytest.approx([3.0])
    assert second["action"].value == pytest.approx([3.0])


def test_naive_async_buffer_skips_steps_elapsed_since_request_start():
    buffer = NaiveAsyncBuffer(chunk_size=3, state_dim=1)
    buffer.add_chunk([[0.0], [1.0], [2.0]], start_timestep=0, chunk_id=1)
    assert buffer.pop_next_action()["action"].tolist() == [0.0]
    assert buffer.pop_next_action()["action"].tolist() == [1.0]

    buffer.add_chunk([[10.0], [11.0], [12.0]], start_timestep=1, chunk_id=2)
    action = buffer.pop_next_action()

    assert action["chunk_id"] == 2
    assert action["chunk_step_index"] == 1
    assert action["action"].tolist() == [11.0]


def test_naive_async_buffer_aligns_ttrtc_reply_to_request_snapshot():
    buffer = NaiveAsyncBuffer(chunk_size=5, state_dim=1, smooth_method="raw")
    first_model_chunk = np.arange(5, dtype=np.float32)[:, None]
    assert buffer.integrate_new_chunk(first_model_chunk, max_k=0, actions_model_chunk=first_model_chunk) is not None
    request_remaining = buffer.get_chunk_progress()["remaining_steps"]

    assert buffer.pop_next_action()["action"] == pytest.approx([0.0])
    assert buffer.pop_next_action()["action"] == pytest.approx([1.0])

    next_model_chunk = (100 + np.arange(5, dtype=np.float32))[:, None]
    switch = buffer.integrate_new_chunk(
        next_model_chunk,
        max_k=0,
        actions_model_chunk=next_model_chunk,
        drop_reference_remaining=request_remaining,
    )

    assert switch is not None
    assert switch["dropped_new_chunk_steps"] == 2
    assert buffer.pop_next_action()["action"] == pytest.approx([102.0])
    assert buffer.get_prev_action_chunk_model() == pytest.approx(next_model_chunk)


def test_traceable_buffer_aligns_ttrtc_reply_and_keeps_last_integrated_reference():
    buffer = StreamActionBuffer(smooth_method="raw")
    first_model_chunk = np.arange(5, dtype=np.float32)[:, None]
    assert buffer.integrate_new_chunk(first_model_chunk, max_k=0, actions_model_chunk=first_model_chunk) is not None
    request_remaining = buffer.get_chunk_progress()["remaining_steps"]
    assert buffer.pop_next_action() is not None
    assert buffer.pop_next_action() is not None

    next_model_chunk = (100 + np.arange(5, dtype=np.float32))[:, None]
    switch = buffer.integrate_new_chunk(
        next_model_chunk,
        max_k=0,
        actions_model_chunk=next_model_chunk,
        drop_reference_remaining=request_remaining,
    )
    assert switch is not None
    assert switch["dropped_new_chunk_steps"] == 2
    assert buffer.pop_next_action()["action"].value == pytest.approx([102.0])

    too_late_chunk = (200 + np.arange(5, dtype=np.float32))[:, None]
    assert buffer.integrate_new_chunk(
        too_late_chunk,
        max_k=0,
        actions_model_chunk=too_late_chunk,
        drop_reference_remaining=100,
    ) is None
    assert buffer.get_prev_action_chunk_model() == pytest.approx(next_model_chunk)


def test_temporal_ensembling_buffer_aggregates_predictions_by_timestep():
    buffer = TemporalEnsemblingBuffer(chunk_size=3, state_dim=1, exp_weight_m=0.0)
    buffer.add_chunk([[0.0], [10.0], [20.0]], start_timestep=0, chunk_id=1)
    buffer.add_chunk([[100.0], [110.0], [120.0]], start_timestep=0, chunk_id=2)

    first = buffer.pop_next_action()
    second = buffer.pop_next_action()

    assert first["chunk_step_index"] == 0
    assert first["action"].tolist() == pytest.approx([50.0])
    assert second["chunk_step_index"] == 1
    assert second["action"].tolist() == pytest.approx([60.0])


def test_create_action_buffer_selects_async_buffer_implementations():
    naive = create_action_buffer({}, max_chunks=10, state_dim=1, smooth_method="raw")
    ensemble = create_action_buffer(
        {"exp_weight_m": 0.02},
        max_chunks=10,
        state_dim=1,
        smooth_method="temporal_ensembling",
    )

    assert isinstance(naive, NaiveAsyncBuffer)
    assert isinstance(ensemble, TemporalEnsemblingBuffer)
    assert ensemble.exp_weight_m == pytest.approx(0.02)


def test_stream_action_buffer_uses_traceable_action_chunk_storage():
    buffer = StreamActionBuffer(smooth_method="raw")
    assert isinstance(buffer.cur_chunk, TraceableActionChunk)

    assert buffer.integrate_new_chunk([[1.0], [2.0]], max_k=0, min_m=1) is not None
    assert isinstance(buffer.cur_chunk, TraceableActionChunk)

    assert buffer.pop_next_action() is not None
    assert isinstance(buffer.cur_chunk, TraceableActionChunk)


def test_stream_action_buffer_tracks_value_sources_through_temporal_smoothing():
    buffer = StreamActionBuffer(smooth_method="temporal_smoothing")
    assert buffer.integrate_new_chunk([[0.0], [10.0]], max_k=0, min_m=2) is not None
    assert buffer.integrate_new_chunk([[100.0], [110.0], [120.0]], max_k=0, min_m=2) is not None

    assert buffer.get_current_action_value_sources() == [
        ((src(1, 0, 0.0, 1.0), src(2, 0, 100.0, 0.0)),),
        ((src(1, 1, 10.0, 0.0), src(2, 1, 110.0, 1.0)),),
        ((src(2, 2, 120.0),),),
    ]
    first = buffer.pop_next_action()

    assert first is not None
    assert "action_source" not in first
    assert "action_value_source" not in first
    assert isinstance(first["action"], TraceableAction)
    assert first["action"].value_source == ((src(1, 0, 0.0, 1.0), src(2, 0, 100.0, 0.0)),)


def test_stream_action_buffer_prints_source_map_before_integrate_returns(capsys):
    buffer = StreamActionBuffer(smooth_method="temporal_smoothing")
    assert buffer.integrate_new_chunk([[0.0], [10.0], [20.0], [30.0], [40.0]], max_k=0, min_m=3) is not None
    _ = capsys.readouterr()

    assert buffer.integrate_new_chunk([[100.0], [110.0], [120.0], [130.0], [140.0]], max_k=0, min_m=3) is not None

    captured = capsys.readouterr()
    assert captured.out == " 1  1  1       \n 2  2  2  2  2 \n □  □  □  □  □ \n"
    assert buffer.format_source_map() == "\n".join(
        [
            " 1  1  1       ",
            " 2  2  2  2  2 ",
            " □  □  □  □  □ ",
        ]
    )


def test_stream_action_buffer_tracks_value_sources_in_raw_mode():
    buffer = StreamActionBuffer(smooth_method="raw")
    assert buffer.integrate_new_chunk([[1.0, 2.0], [3.0, 4.0]], max_k=0, min_m=2) is not None

    assert buffer.get_current_action_value_sources() == [
        ((src(1, 0, 1.0),), (src(1, 0, 2.0),)),
        ((src(1, 1, 3.0),), (src(1, 1, 4.0),)),
    ]
    first = buffer.pop_next_action()

    assert first is not None
    assert "action_source" not in first
    assert "action_value_source" not in first
    assert isinstance(first["action"], TraceableAction)
    assert first["action"].value_source == ((src(1, 0, 1.0),), (src(1, 0, 2.0),))


def test_stream_action_buffer_can_use_request_id_as_chunk_id():
    buffer = StreamActionBuffer(smooth_method="raw")
    assert buffer.integrate_new_chunk([[1.0, 2.0]], max_k=0, min_m=1, chunk_id=42) is not None

    first = buffer.pop_next_action()

    assert first is not None
    assert first["chunk_id"] == 42
    assert first["chunk_step_index"] == 0
    assert "action_value_source" not in first
    assert isinstance(first["action"], TraceableAction)
    assert first["action"].value_source == ((src(42, 0, 1.0),), (src(42, 0, 2.0),))


def test_stream_action_buffer_tracks_per_value_sources_through_temporal_smoothing():
    buffer = StreamActionBuffer(smooth_method="temporal_smoothing")
    assert buffer.integrate_new_chunk([[0.0, 10.0], [20.0, 30.0]], max_k=0, min_m=2) is not None
    assert buffer.integrate_new_chunk([[100.0, 110.0], [120.0, 130.0]], max_k=0, min_m=2) is not None

    first = buffer.pop_next_action()
    second = buffer.pop_next_action()

    assert first is not None
    assert second is not None
    assert isinstance(first["action"], TraceableAction)
    assert isinstance(second["action"], TraceableAction)
    assert "action_value_source" not in first
    assert "action_value_source" not in second
    assert first["action"].value_source == (
        (src(1, 0, 0.0, 1.0), src(2, 0, 100.0, 0.0)),
        (src(1, 0, 10.0, 1.0), src(2, 0, 110.0, 0.0)),
    )
    assert second["action"].value_source == (
        (src(1, 1, 20.0, 0.0), src(2, 1, 120.0, 1.0)),
        (src(1, 1, 30.0, 0.0), src(2, 1, 130.0, 1.0)),
    )


def test_stream_action_buffer_tracks_value_sources_after_dropping_new_prefix():
    buffer = StreamActionBuffer(smooth_method="temporal_smoothing")
    assert buffer.integrate_new_chunk([[0.0], [10.0], [20.0]], max_k=0, min_m=2) is not None
    assert buffer.pop_next_action() is not None
    assert buffer.pop_next_action() is not None
    assert buffer.integrate_new_chunk([[100.0], [110.0], [120.0], [130.0]], max_k=2, min_m=2) is not None

    assert buffer.get_current_action_value_sources() == [
        ((src(1, 2, 20.0, 1.0), src(2, 2, 120.0, 0.0)),),
        ((src(1, 2, 20.0, 0.0), src(2, 3, 130.0, 1.0)),),
    ]


def test_traceable_action_chunk_timing_smoke():
    actions = np.arange(50 * 14, dtype=float).reshape(50, 14)
    loop_count = 200

    start = time.perf_counter()
    for i in range(loop_count):
        b = TraceableActionChunk.from_array(actions, chunk_id=10 + i * 3)
        c = TraceableActionChunk.from_array(actions + 1.0, chunk_id=11 + i * 3)
        a = TraceableActionChunk.zeros(actions.shape, chunk_id=12 + i * 3)
        a[4:44] = b[4:44] + c[4:44]
        TraceableActionChunk.weighted_sum(((0.4, b[0:40]), (0.6, c[0:40])))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    per_loop_ms = elapsed_ms / loop_count

    print(f"traceable_action_chunk loop_count={loop_count} total_ms={elapsed_ms:.3f} per_loop_ms={per_loop_ms:.3f}")
    assert per_loop_ms < 10.0
