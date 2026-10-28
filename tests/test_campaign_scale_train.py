import pytest
from scripts.campaign_scale_train import dimensions, selected_devices


def test_equal_environment_budget_reports_changed_optimizer_reuse():
    base=dict(devices=8,envs_per_device=16,rollout_ticks=128,global_minibatch=512)
    assert dimensions(base)==(16384,64,32)
    assert dimensions(dict(base,envs_per_device=32))==(32768,64,64)
    assert dimensions(dict(base,global_minibatch=256))==(16384,32,64)
    assert 5242880//16384==320 and 5242880//32768==160


def test_multicard_global_batch_held_constant():
    for devices in (1,2,4,8):
        assert dimensions(dict(devices=devices,envs_per_device=128//devices,
            rollout_ticks=128,global_minibatch=512))==(16384,512//devices,32)


def test_invalid_partition_rejected():
    with pytest.raises(ValueError):
        dimensions(dict(devices=8,envs_per_device=3,rollout_ticks=17,global_minibatch=512))


def test_single_device_override_selects_physical_gpu_without_changing_multidevice_default():
    assert selected_devices(dict(devices=1), "6") == "6"
    assert selected_devices(dict(devices=8)) == "0,1,2,3,4,5,6,7"
    with pytest.raises(ValueError):
        selected_devices(dict(devices=8), "6")
