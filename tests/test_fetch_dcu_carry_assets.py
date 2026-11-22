"""Input installation must reject wrong weights before they can be loaded."""
import hashlib
import io
import urllib.error

import pytest

from scripts import fetch_dcu_carry_assets as assets


@pytest.fixture
def payload(tmp_path, monkeypatch):
    data = b'verified-test-weights'
    source = tmp_path / 'source.pt'
    source.write_bytes(data)
    monkeypatch.setattr(assets, 'EXPECTED_BYTES', len(data))
    monkeypatch.setattr(assets, 'EXPECTED_SHA', hashlib.sha256(data).hexdigest())
    return source, tmp_path / 'installed' / 'actor.pt'


def test_reject_wrong_model_without_publishing(payload):
    source, destination = payload
    source.write_bytes(b'X' * source.stat().st_size)
    with pytest.raises(ValueError, match='SHA256'):
        assets.install(source, destination)
    assert not destination.exists()


def test_install_is_atomic_and_idempotent(payload):
    source, destination = payload
    assert assets.install(source, destination) == 'installed'
    assert destination.read_bytes() == source.read_bytes()
    assert assets.install(source, destination) == 'already_verified'
    assert list(destination.parent.iterdir()) == [destination]


def test_existing_invalid_model_is_preserved(payload):
    source, destination = payload
    destination.parent.mkdir()
    destination.write_bytes(b'wrong-existing-file')
    with pytest.raises(ValueError):
        assets.install(source, destination)
    assert destination.read_bytes() == b'wrong-existing-file'


def test_destination_symlink_is_rejected(payload):
    source, destination = payload
    destination.parent.mkdir()
    destination.symlink_to(source)
    with pytest.raises(ValueError, match='linked'):
        assets.install(source, destination)
    assert destination.is_symlink()


def test_download_falls_back_after_proxy_failure(payload, monkeypatch):
    source, destination = payload
    calls = []
    def incoming(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib.error.URLError('proxy unavailable')
        return io.BytesIO(source.read_bytes())
    monkeypatch.setattr(assets.urllib.request, 'urlopen', incoming)
    assert assets.install(destination=destination) == 'installed'
    assert destination.read_bytes() == source.read_bytes()
    assert calls == [assets.URL, assets.API_URL]


def test_bad_downloads_never_publish(payload, monkeypatch):
    _, destination = payload
    monkeypatch.setattr(assets.urllib.request, 'urlopen',
                        lambda *a, **k: io.BytesIO(b'invalid response'))
    with pytest.raises(ValueError, match='SHA256'):
        assets.install(destination=destination)
    assert not destination.exists()
    assert not list(destination.parent.glob('.u930-*'))
