"""Verify exact Pages commit, catalog identity, every new model/replay byte."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

def verify(base, expected):
    def fetch(name):
        req=urllib.request.Request(base+name+'?je10='+expected,headers={'Cache-Control':'no-cache','User-Agent':'QQT-JE10-verifier'})
        with urllib.request.urlopen(req,timeout=180) as res:
            if res.status!=200:raise ValueError('HTTP '+str(res.status))
            return res.read()
    build=json.loads(fetch('build-info.json'))
    if build['commit']!=expected:raise ValueError('Pages head mismatch')
    local=json.loads(Path('web/review-catalog.json').read_text());remote=json.loads(fetch('review-catalog.json'))
    rows=local['replays'][:4]
    if remote['replays'][:4]!=rows:raise ValueError('online first four Review rows differ')
    manifest=json.loads(fetch('models.json'));model=json.loads(Path('web/models.json').read_text())['models'][0]
    if manifest['models'][0]!=model:raise ValueError('online model selector first row differs')
    result=dict(passed=True,commit=expected,base=base,first_review=rows[0]['id'],checkpoint_sha256=model['source_sha256'],assets=[])
    assets=[('models/'+a['file'],a) for a in [model,model['onnx']]]+ [('review-replays/'+r['file'],r) for r in rows]
    for name,row in assets:
        data=fetch(name);digest=hashlib.sha256(data).hexdigest()
        if len(data)!=row['bytes'] or digest!=row['sha256']:raise ValueError('online hash/length mismatch '+name)
        if name.endswith('.json'):
            d=json.loads(data);cp=d['meta']['checkpoint_sha256']
            if cp!=model['source_sha256']:raise ValueError('checkpoint identity')
            if name.startswith('review-') and (d['meta']['model_sha256']!=model['sha256'] or d['meta']['onnx_sha256']!=model['onnx']['sha256']):raise ValueError('replay model identity')
        result['assets'].append(dict(path=name,http=200,bytes=len(data),sha256=digest))
        print('HTTP 200 SHA-256 GREEN',name,flush=True)
    return result
if __name__=='__main__':
    result=verify(sys.argv[1],sys.argv[2]);Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+'\n');print('Exact Pages head and all release assets GREEN')
