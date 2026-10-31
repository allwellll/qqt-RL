"""Authenticated release/Actions operations; credentials stay in process memory."""
import json,subprocess,urllib.request,urllib.error,sys,time,os
from pathlib import Path
O=Path(os.environ.get('BT_PUBLICATION_BASE','runs/bomb_threat_v1/publish_a3sr_u1280_20261030'))

def request(path,method='GET',data=None,ctype='application/json'):
    answer=subprocess.check_output(['git','credential','fill'],input=b'protocol=https\nhost=github.com\n\n',stderr=subprocess.DEVNULL)
    credential=dict(l.split('=',1) for l in answer.decode().splitlines() if '=' in l)
    req=urllib.request.Request(path if path.startswith('https:') else 'https://api.github.com/repos/allwellll/qqt-RL'+('/'+path if path else ''),
        method=method,data=data,headers={'Authorization':'Bearer '+credential['password'],'Accept':'application/vnd.github+json','Content-Type':ctype,'User-Agent':'qqt-release-verification','X-GitHub-Api-Version':'2022-11-28'})
    with urllib.request.build_opener().open(req,timeout=90) as response:return json.load(response)

def main():
    mode=sys.argv[1];tag=os.environ.get('BT_RELEASE_TAG','web-bun392-a3sr-u1280-20261030-r1')
    if mode=='probe':
        r=request('');print(json.dumps({'authenticated':True,'repo':r['full_name'],'push':r['permissions']['push']}));return
    if mode=='release':
        try:r=request('releases/tags/'+tag)
        except urllib.error.HTTPError as e:
            if e.code!=404:raise
            body=(O/'RELEASE.md').read_text()
            expected=json.loads((O/'manifest.json').read_text())
            assert expected['tag']==tag
            # Never let Releases silently create a tag on the old default HEAD.
            bound=request('git/ref/tags/'+tag)
            assert bound['object']['sha']==json.loads((O/'GIT_OBJECT_IMPORT.json').read_text())['tag_object']
            r=request('releases','POST',json.dumps(dict(tag_name=tag,name=json.loads((O/'CATALOG_ROW.json').read_text())['onnx']['meta']['name']+' · stock-window-v1',body=body,draft=False,prerelease=False)).encode())
        manifest=json.loads((O/'manifest.json').read_text());existing={a['name']:a for a in r['assets']}
        names=[a['file'] for a in manifest['assets']]+['manifest.json','SHA256SUMS']
        for name in names:
            if name not in existing:
                url=r['upload_url'].split('{')[0]+'?name='+name
                existing[name]=request(url,'POST',(O/name).read_bytes(),'application/octet-stream')
        for asset in manifest['assets']:
            a=existing[asset['file']];assert a['size']==asset['bytes']
            if a.get('digest'):assert a['digest']=='sha256:'+asset['sha256']
        record=dict(status='RELEASE_ASSETS_UPLOADED',url=r['html_url'],tag=tag,assets=[dict(name=a['name'],bytes=a['size'],digest=a.get('digest')) for a in existing.values()],unix=time.time())
        (O/'RELEASE_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));return
    if mode=='actions':
        head=sys.argv[2];r=request('actions/runs?head_sha='+head+'&per_page=20')
        runs=[{k:x[k] for k in ['id','name','status','conclusion','html_url','head_sha']} for x in r['workflow_runs']]
        (O/'ACTIONS.json').write_text(json.dumps(dict(head=head,runs=runs,unix=time.time()),indent=2)+'\n');print(json.dumps(runs));return
if __name__=='__main__':main()
