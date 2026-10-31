from scripts.expert_validation_summarize import paired_delivery_comparison


def test_paired_delivery_counts_separate_both_success_and_both_failure():
    left = {
        10: {"delivery": 1},
        11: {"delivery": 0},
        12: {"delivery": 1},
        13: {"delivery": 0},
    }
    right = {
        10: {"delivery": 0},
        11: {"delivery": 1},
        12: {"delivery": 1},
        13: {"delivery": 0},
    }

    result = paired_delivery_comparison(left, right, seed=7)

    assert result["paired_games"] == 4
    assert result["left_only_deliveries"] == 1
    assert result["right_only_deliveries"] == 1
    assert result["both_delivered"] == 1
    assert result["neither_delivered"] == 1
    assert result["mean_paired_delivery_difference"] == 0
