"""Assemble same-origin Pages models from the catalog's versioned Releases."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request


def download_models(manifest, output, fetch=None):
    if manifest.get('schema') != 'qqt.web-models/v1':
        raise ValueError('invalid catalog schema')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    def network(url):
        with urllib.request.urlopen(url, timeout=120) as response:
            return response.read()
    fetch = fetch or network
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
