import hashlib
import pytest
from scripts.fetch_web_models import download_models


def row():
    return dict(id='test',file='test.json',bytes=2,sha256=hashlib.sha256(b'{}').hexdigest(),
                url='https://github.com/example/project/releases/download/v1/test.json')


def test_all_catalog_releases_download_and_verify(tmp_path):
    download_models({'schema':'qqt.web-models/v1','models':[row()]},tmp_path,lambda url:b'{}')
    assert (tmp_path/'test.json').read_bytes() == b'{}'


def test_corrupt_download_never_replaces_model(tmp_path):
    (tmp_path/'test.json').write_bytes(b'previous')
    with pytest.raises(ValueError,match='integrity'):
        download_models({'schema':'qqt.web-models/v1','models':[row()]},tmp_path,lambda url:b'xx')
    assert (tmp_path/'test.json').read_bytes() == b'previous'


def test_unsafe_file_and_nonrelease_url_rejected(tmp_path):
    for changes in [dict(file='../test.json'),dict(url='https://example.com/test.json')]:
        with pytest.raises(ValueError):
            download_models({'schema':'qqt.web-models/v1','models':[{**row(),**changes}]},tmp_path)
