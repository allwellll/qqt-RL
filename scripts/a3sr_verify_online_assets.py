"""Verify live Pages manifest/rules and complete same-origin model bytes."""
import concurrent.futures,json,urllib.request,time,hashlib,os
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
O=Path(os.environ.get('BT_PUBLICATION_BASE','runs/bomb_threat_v1/publish_a3sr_u1280_20261030'));BASE='https://allwellll.github.io/qqt-RL/'
def get(url,headers=None):
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers or {}),timeout=120) as r:return r.status,dict(r.headers),r.read()
def main():
    row=json.loads((O/'CATALOG_ROW.json').read_text());head=json.loads((O/'MAIN_PUSH.json').read_text())['head'];stamp=str(int(time.time()))
    _,_,build=get(BASE+'build-info.json?v='+stamp);assert json.loads(build)['commit']==head
    _,_,manifest=get(BASE+'models.json?v='+stamp);local=(Path(os.environ.get('BT_PUBLICATION_WORKTREE','../cx_main_integration_20261028'))/'web/models.json').read_bytes();assert manifest==local
    _,_,sim=get(BASE+'sim-stock-window-v1.js?v='+stamp);assert hashlib.sha256(sim).hexdigest()=='2767cb4e2993383addc72d792fef8947f3ff86e1dcad4f73c8e1c04752a13f45'
    receipts=[]
    for asset in (row,row['onnx']):
        n=asset['bytes'];target=O/('online-'+asset['file']);size=1024*1024
        def chunk(start):
            end=min(n-1,start+size-1)
            for attempt in range(3):
                try:
                    code,h,b=get(BASE+'models/'+asset['file']+f'?verify={stamp}-{start}-{attempt}',{'Range':f'bytes={start}-{end}'})
                    if code==200 and len(b)==n:return start,b[start:end+1]
                    assert code==206 and h.get('Content-Range',h.get('content-range'))==f'bytes {start}-{end}/{n}',(code,len(b))
                    assert len(b)==end-start+1
                    return start,b
                except Exception:
                    if attempt==2:raise
            raise RuntimeError('download exhausted')
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            pieces=sorted(pool.map(chunk,range(0,n,size)))
        data=b''.join(b for _,b in pieces);assert len(data)==n and hashlib.sha256(data).hexdigest()==asset['sha256']
        target.write_bytes(data);receipts.append(dict(file=asset['file'],bytes=n,sha256=asset['sha256'],http_range_verified=True))
        print('complete Pages bytes verified',asset['file'],flush=True)
    atomic_json(O/'ONLINE_ASSETS.json',dict(status='PASS',head=head,build=json.loads(build),models_sha256=hashlib.sha256(manifest).hexdigest(),sim_sha256=hashlib.sha256(sim).hexdigest(),assets=receipts,url=BASE,unix=time.time()))
if __name__=='__main__':main()
