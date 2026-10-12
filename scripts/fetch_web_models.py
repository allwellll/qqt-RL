"""Assemble same-origin Pages models from the catalog's versioned Releases."""
import argparse
from datetime import timezone
from email.utils import parsedate_to_datetime
import hashlib
import http.client
import json
from pathlib import Path
import re
import ssl
import time
import urllib.parse
import urllib.error
import urllib.request


def retry_after_seconds(value, now):
    if not value:
        return 0
    try:
        if re.fullmatch(r'\d+', value.strip()):
            return int(value)
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return max(0, date.timestamp() - now())
    except (ValueError, TypeError, OverflowError):
        return 0


def fetch_bytes(url, opener=urllib.request.urlopen, pause=time.sleep, attempts=5,
                now=time.time, max_delay=60):
    """At most five requests; 1/2/4/8s backoff or the longer Retry-After.

    A server delay above the wait budget fails rather than retrying too early.
    Unknown URL, TLS certificate, permanent HTTP and integrity errors propagate.
    """
    if not 1 <= attempts <= 5 or not 1 <= max_delay <= 60:
        raise ValueError('retry attempts/delay exceed bounded policy')
    request = urllib.request.Request(url, headers={'User-Agent':'QQT-web-models/1'})
    for attempt in range(attempts):
        delay = 2**attempt
        try:
            with opener(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt+1 == attempts:
                raise
            delay = max(delay, retry_after_seconds((error.headers or {}).get('Retry-After'), now))
            if delay > max_delay:
                raise
            error.close()
            print(f'temporary model HTTP {error.code}; retry {attempt+1}/{attempts}', flush=True)
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.IncompleteRead, ssl.SSLEOFError) as error:
            reason = error.reason if isinstance(error, urllib.error.URLError) else error
            if not isinstance(reason, (TimeoutError, ConnectionError,
                                       http.client.IncompleteRead, ssl.SSLEOFError)) or attempt+1 == attempts or delay > max_delay:
                raise
            print(f'temporary model transport error; retry {attempt+1}/{attempts}', flush=True)
        pause(delay)


def download_models(manifest, output, fetch=None):
    if manifest.get('schema') != 'qqt.web-models/v1':
        raise ValueError('invalid catalog schema')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fetch = fetch or fetch_bytes
    seen = set()
    for row in manifest['models']:
        name = row['file']; url = urllib.parse.urlparse(row['url'])
        if not re.fullmatch(r'[a-z0-9][a-z0-9._-]*\.json', name) or name in seen:
            raise ValueError('unsafe or duplicate model filename')
        seen.add(name)
        if (url.scheme != 'https' or url.netloc != 'github.com' or url.query or url.fragment
                or not re.fullmatch(r'/[^/]+/[^/]+/releases/download/[^/]+/'+re.escape(name),url.path)):
            raise ValueError('model must use an HTTPS GitHub Release asset URL')
        data = fetch(row['url'])
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f"model integrity failed: {row['id']}")
        target = output/name; pending = target.with_suffix('.tmp')
        pending.write_bytes(data); pending.replace(target)
    return len(seen)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('web/models.json'))
    p.add_argument('--output', type=Path, default=Path('_site/models'))
    a = p.parse_args()
    print(f"verified {download_models(json.loads(a.manifest.read_text()), a.output)} same-origin models")
