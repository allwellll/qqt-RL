import numpy as np

from qqt_rl.training.counterfactual import (
    StateRequest,
    build_state_requests,
    compare_replay_arrays,
    padded_batches,
)


def test_state_requests_preserve_split_seed_and_tick_order():
    requests = build_state_requests(
        seed_base=1000, split_stride=100,
        split_counts={"train": 3, "validation": 2, "test": 1})
    assert [(row.split, row.seed, row.ticks) for row in requests] == [
        ("train", 1000, 8), ("train", 1001, 24), ("train", 1002, 48),
        ("validation", 1100, 8), ("validation", 1101, 24),
        ("test", 1200, 8),
    ]


def test_padded_batches_keep_fixed_shape_without_declaring_padding():
    rows = [StateRequest("train", index, 8) for index in range(5)]
    batches = list(padded_batches(rows, 4))
    assert len(batches) == 2
    assert all(len(batch.requests) == 4 for batch in batches)
    assert batches[0].valid_count == 4
    assert batches[1].valid_count == 1
    assert batches[1].requests[0].seed == 4
    assert [row.seed for row in batches[1].requests[1:]] == [4, 4, 4]


def test_replay_comparison_reports_numeric_error_and_exact_seed_failures():
    reference = {
        "declared_seed": np.asarray([1, 2], np.int64),
        "q": np.asarray([[1.0, 2.0]], np.float32),
        "split_name": np.asarray(["train", "test"]),
    }
    candidate = {
        "declared_seed": np.asarray([1, 3], np.int64),
        "q": np.asarray([[1.0, 2.125]], np.float32),
        "split_name": np.asarray(["train", "test"]),
    }
    report = compare_replay_arrays(reference, candidate)
    assert report["compatible"] is False
    assert report["fields"]["declared_seed"]["exact"] is False
    assert report["fields"]["q"]["max_abs_error"] == 0.125
    assert report["fields"]["split_name"]["exact"] is True
