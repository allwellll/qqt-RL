"""R5 opt-in native export with visible-input features inside ONNX.

Legacy exporters and the R4 runtime remain untouched. Raw deployment inputs
stay obs24/state24; both ONNX and the JS add-on derive the same five fields.
"""
from pathlib import Path
import pickle
import torch
from torch.nn import functional as F
from scripts.je03_native_vit_export_20261011 import NativeViT


def features(obs, state):
    home = obs[:, 14:15] > 0
    padded = F.pad(home, (1, 1, 1, 1))
    halo = (home | padded[:, :, :-2, 1:-1] | padded[:, :, 2:, 1:-1]
            | padded[:, :, 1:-1, :-2] | padded[:, :, 1:-1, 2:])
    near = (obs[:, 0:1] * halo).sum((1, 2, 3)) >= .5
    cargo = (state[:, 9] > 0) & (state[:, 14] > 0)
    free_empty = (state[:, 9] == 1) & (state[:, 13] == 0)
    drop = free_empty & (obs[:, 17] > 0).flatten(1).any(-1)
    stock = free_empty & ((state[:, 10] <= 0) | (state[:, 10] > 1)) & (state[:, 20] > 0)
    # Stack explicit booleans rather than introducing a phase prediction head.
    return torch.stack([~cargo & ~drop & ~stock, ~cargo & ~drop & stock,
                        ~cargo & drop, cargo & ~near, cargo & near], -1).to(state.dtype)


class PhaseViT(NativeViT):
    def __init__(self, params):
        super().__init__(params)
        self.conditioned = params['state_w'].shape[0] == 29

    def forward(self, obs, state):
        if self.conditioned:
            state = torch.cat([state, features(obs, state)], -1)
        return super().forward(obs, state)


def export(source, target):
    torch.set_num_threads(1)
    with Path(source).open('rb') as stream:
        params = pickle.load(stream)
    model = PhaseViT(params).eval()
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, (torch.zeros((1, 24, 13, 15)), torch.zeros((1, 24))),
                          str(target) + '.tmp', input_names=['obs', 'state'],
                          output_names=['move', 'bomb', 'value'], opset_version=17, dynamo=False)
    Path(str(target) + '.tmp').replace(target)


def authority_export(policy, out):
    """R5-only authority callback; never patch the shared exporter on disk."""
    import subprocess
    import sys
    from scripts.population_run_state import atomic_json, sha
    root = Path(__file__).resolve().parents[1]
    policy, out = Path(policy).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    target, onnx = out/'model.json', out/'model.onnx'
    if not target.exists():
        subprocess.run([sys.executable, 'scripts/export_bun_web_model.py', str(policy),
                        '--output', str(target.with_suffix('.unsealed.json'))], check=True)
        export(policy, onnx)
        import json
        doc = json.loads(target.with_suffix('.unsealed.json').read_text())
        with Path(policy).open('rb') as stream:
            conditioned = pickle.load(stream)['state_w'].shape[0] == 29
        doc['meta'].update(bun_state_encoding='bun24-trap-timer-v2',
                           onnx_path=str(onnx.relative_to(root)), onnx_sha256=sha(onnx),
                           checkpoint_sha256=sha(policy))
        if conditioned:
            doc['meta'].update(phase_conditioning='phase5-visible-v1',
                               phase_runtime='web/retention_phase_conditioning.js')
        atomic_json(target, doc)
    import json
    doc = json.loads(target.read_text())
    assert doc['meta']['checkpoint_sha256'] == sha(policy)
    assert doc['meta']['onnx_sha256'] == sha(onnx)
    return target
