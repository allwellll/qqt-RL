import base64
from pathlib import Path

import numpy as np

from scripts.export_bun_web_model import extract_transformer, pack_tensors


def test_transformer_export_schema_round_trip():
    linear = lambda rows, cols: (np.arange(rows * cols).reshape(rows, cols), np.zeros(cols))
    block = {
        "ln1_g": np.ones(4), "ln1_b": np.zeros(4),
        "ln2_g": np.ones(4), "ln2_b": np.zeros(4),
        "q": linear(4, 4), "k": linear(4, 4), "v": linear(4, 4),
        "proj": linear(4, 4), "ff1": linear(4, 8), "ff2": linear(8, 4),
    }
    params = {
        "tok": linear(24 * 9, 4), "pos": (np.zeros((1, 31, 4)),),
        "state_w": np.zeros((24, 4)), "state_b": np.zeros(4),
        "blocks": [block],
        "heads": {"wm": linear(4, 5), "wb": linear(4, 3), "wv": linear(4, 128)},
    }
    tensors = extract_transformer(params)
    encoded, index = pack_tensors(tensors)
    decoded = np.frombuffer(base64.b64decode(encoded), dtype=np.float32)
    assert set(index) == set(tensors)
    assert decoded.size == sum(value.size for value in tensors.values())
    assert index["head_wm_w"][1] == 20


def test_web_runtime_uses_bun_action_and_value_metadata():
    root = Path(__file__).resolve().parents[1]
    exporter = (root / "scripts/export_bun_web_model.py").read_text()
    runtime = (root / "web/sim.js").read_text()
    assert '"ability_actions": 3' in exporter
    assert '"value_min": -20.0, "value_max": 20.0' in exporter
    assert "this.meta.value_min" in runtime
    assert "this.meta.value_max" in runtime
