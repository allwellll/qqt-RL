import hashlib
import pytest
from scripts.fetch_web_models import download_models, fetch_bytes


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


def test_transient_release_http_error_retried():
    import io
    import urllib.error
    calls=[];pauses=[]
    def open_url(request, timeout):
        calls.append(request)
        if len(calls)==1:
            raise urllib.error.HTTPError(request.full_url,500,'temporary',{},None)
        return io.BytesIO(b'{}')
    assert fetch_bytes(row()['url'],opener=open_url,pause=pauses.append)==b'{}'
    assert len(calls)==2 and pauses==[1]
    assert calls[0].get_header('User-agent')=='QQT-web-models/1'


def test_permanent_http_error_not_retried():
    import urllib.error
    calls=[]
    def open_url(request, timeout):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url,404,'missing',{},None)
    with pytest.raises(urllib.error.HTTPError):
        fetch_bytes(row()['url'],opener=open_url,pause=lambda _:None)
    assert len(calls)==1


def test_continuous_500_exhausts_bound_and_never_skips_model(tmp_path):
    import urllib.error
    calls=[];pauses=[]
    def open_url(request, timeout):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url,500,'temporary',{},None)
    with pytest.raises(urllib.error.HTTPError):
        download_models({'schema':'qqt.web-models/v1','models':[row()]},tmp_path,
            lambda url:fetch_bytes(url,opener=open_url,pause=pauses.append))
    assert len(calls)==5 and pauses==[1,2,4,8]
    assert not (tmp_path/'test.json').exists()


@pytest.mark.parametrize('header,expected',[
    ('7',7), ('Thu, 01 Jan 1970 00:01:40 GMT',10), ('invalid',1), ('0',1)])
def test_retry_after_respected_for_seconds_and_http_date(header,expected):
    import io
    import urllib.error
    calls=[];pauses=[]
    def open_url(request,timeout):
        calls.append(request)
        if len(calls)==1:
            raise urllib.error.HTTPError(request.full_url,429,'temporary',{'Retry-After':header},None)
        return io.BytesIO(b'{}')
    assert fetch_bytes(row()['url'],opener=open_url,pause=pauses.append,now=lambda:90)==b'{}'
    assert pauses==[expected]


def test_retry_after_beyond_wait_bound_fails_instead_of_retrying_early():
    import urllib.error
    pauses=[];calls=[]
    def open_url(request,timeout):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url,503,'temporary',{'Retry-After':'120'},None)
    with pytest.raises(urllib.error.HTTPError):
        fetch_bytes(row()['url'],opener=open_url,pause=pauses.append)
    assert len(calls)==1 and not pauses


@pytest.mark.parametrize('code',[400,401,403,404,408])
def test_other_http_codes_fail_immediately(code):
    import urllib.error
    calls=[];pauses=[]
    def open_url(request,timeout):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url,code,'permanent',{},None)
    with pytest.raises(urllib.error.HTTPError):
        fetch_bytes(row()['url'],opener=open_url,pause=pauses.append)
    assert len(calls)==1 and not pauses


def test_transport_disconnect_retried_but_unknown_url_error_not_retried():
    import io
    import urllib.error
    calls=[];pauses=[]
    def open_url(request,timeout):
        calls.append(request)
        if len(calls)==1:raise urllib.error.URLError(ConnectionResetError('disconnected'))
        return io.BytesIO(b'{}')
    assert fetch_bytes(row()['url'],opener=open_url,pause=pauses.append)==b'{}'
    assert pauses==[1]
    calls.clear()
    def permanent(request,timeout):
        calls.append(request);raise urllib.error.URLError('unknown URL failure')
    with pytest.raises(urllib.error.URLError):
        fetch_bytes(row()['url'],opener=permanent,pause=lambda _:None)
    assert len(calls)==1
