"""Transparent public Pages transport for CPU browser validation in this container."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

url, directory, headers = sys.argv[1:]
if not url.startswith('https://allwellll.github.io/qqt-RL/'):
    raise ValueError('Only this public Pages deployment is allowed')
headers = json.loads(headers)
cache = Path(directory)
cache.mkdir(parents=True, exist_ok=True)
target = cache / hashlib.sha256(url.encode()).hexdigest()


def open_retry(address, request_headers, method='GET'):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(
                address, headers=request_headers, method=method), timeout=20)
        except Exception:
            if attempt == 3:
                raise


is_range = any(key.lower() == 'range' for key in headers)
# Pages/proxies may ignore Range on HEAD. For ranges, GET supplies the actual
# 206 headers; closing after headers avoids re-reading a cached body.
with open_retry(url, headers, 'GET' if is_range else 'HEAD') as response:
    status = response.status
    full_headers = dict(response.headers)
    size = int(response.headers.get('Content-Length', 0))
    returned_headers = {key.lower(): value for key, value in response.headers.items()
                        if key.lower() in ('content-type', 'content-security-policy',
                                           'x-content-type-options', 'cache-control', 'content-range')}
    if is_range and not (target.exists() and size and target.stat().st_size == size):
        data = response.read()
        if size and len(data) != size:
            raise ValueError('Public range truncated')
        target.write_bytes(data)

# Reuse bytes fetched from this same public URL earlier in this validation.
# Revalidate HTTP shape; browser model SHA verification still runs every load.
if not (target.exists() and size and target.stat().st_size == size):
    if status == 200 and size > 1048576 and not headers:
        step = 262144

        def chunk(start):
            end = min(start + step, size) - 1
            for attempt in range(4):
                try:
                    address = url + ('&' if '?' in url else '?') + 'dcu-range=' + str(start)
                    with open_retry(address, {'Range': f'bytes={start}-{end}'}) as response:
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                            raise ValueError('Public response range mismatch')
                        data = response.read(end - start + 1)
                        if len(data) != end - start + 1:
                            raise ValueError('Public response chunk truncated')
                        return data
                except Exception:
                    if attempt == 3:
                        raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            data = b''.join(pool.map(chunk, range(0, size, step)))
        if len(data) != size:
            raise ValueError('Public response size mismatch')
    else:
        with open_retry(url, headers) as response:
            if response.status != status:
                raise ValueError('HEAD/GET status mismatch')
            data = response.read()
        if size and len(data) != size:
            raise ValueError('Public response truncated')
    target.write_bytes(data)

print(json.dumps(dict(status=status, headers=returned_headers, file=str(target))))
