from scripts.benchmark_v2_pipeline import parse_joint_steps_per_second, semantic_equivalence
from scripts.train_bun_critic import padded_index_batches


def test_semantic_equivalence_requires_exact_discrete_fields():
    report = {"fields": {
        "declared_seed": {"exact": True, "max_abs_error": 0.0},
        "legal": {"exact": True},
        "q": {"exact": False, "max_abs_error": 1e-7},
    }}
    result = semantic_equivalence(report, float_atol=1e-6)
    assert result["equivalent"] is True
    report["fields"]["legal"]["exact"] = False
    assert semantic_equivalence(report, float_atol=1e-6)["equivalent"] is False


def test_joint_smoke_parser_reads_reported_throughput():
    assert parse_joint_steps_per_second(
        "warmup done\nFINAL end-to-end sps = 3,237 (16 trainable-agent steps)\n") == 3237.0


def test_critic_tail_batch_is_padded_with_an_explicit_valid_mask():
    batches = list(padded_index_batches([4, 8, 12, 16, 20], 4))
    assert batches[0][0].tolist() == [4, 8, 12, 16]
    assert batches[0][1].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert batches[1][0].tolist() == [20, 20, 20, 20]
    assert batches[1][1].tolist() == [1.0, 0.0, 0.0, 0.0]
