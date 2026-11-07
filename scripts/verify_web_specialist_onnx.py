"""Verify CPU ONNX against JS on the inputs from check_web_specialist_model.js."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def verify(graph, samples_file, output):
    source = json.loads(Path(samples_file).read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
    state_dim = session.get_inputs()[1].shape[-1]
    samples = [dict(obs=[0.] * (25 * 195), state=[0.] * state_dim,
                    js=source["reference_zero"], map="zero", seat=None, tick=0)] + source["samples"]
    errors = []
    for row in samples:
        result = session.run(None, dict(obs=np.asarray(row["obs"], np.float32).reshape(1, 25, 13, 15),
                                       state=np.asarray(row["state"], np.float32).reshape(1, state_dim)))
        err = max(float(np.max(np.abs(actual.reshape(-1) - np.asarray(row["js"][key]))))
                  for actual, key in zip(result, ("move", "bomb", "value")))
        if err >= .002:
            raise ValueError("JS/ONNX numeric parity failed")
        if "masks" in row:
            for actual, key in zip(result[:2], ("move", "bomb")):
                mask = np.asarray(row["masks"][key], bool)
                a = int(np.argmax(np.where(mask, actual.reshape(-1), -np.inf)))
                b = int(np.argmax(np.where(mask, np.asarray(row["js"][key]), -np.inf)))
                if a != b:
                    raise ValueError("legal masked action parity failed")
        errors.append(dict(map=row["map"], seat=row["seat"], tick=row["tick"], max_abs=err))
    receipt = dict(status="PASS", ruleset=source["ruleset"], samples=len(samples),
                   max_abs=max(row["max_abs"] for row in errors), masked_greedy_exact=True,
                   reference_zero=source["reference_zero"], errors=errors,
                   scope="fp32 CPU JS/ONNX integration; not a win-rate or training-bf16 claim")
    Path(output).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", required=True)
    p.add_argument("--samples", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    r = verify(a.onnx, a.samples, a.output)
    print(json.dumps({k: r[k] for k in ("status", "samples", "max_abs", "masked_greedy_exact")}))
