"""Git Data API transport: preserve exact Git objects, only fast-forward refs."""
import subprocess,json,datetime,sys,time,base64,urllib.error,os
from pathlib import Path
from scripts.a3sr_publish_github import request
O=Path(os.environ.get('BT_PUBLICATION_BASE','runs/bomb_threat_v1/publish_a3sr_u1280_20261030'));W=Path(os.environ.get('BT_PUBLICATION_WORKTREE','../cx_main_integration_20261028'))
def git(*args):return subprocess.check_output(['git','-C',str(W),*args])
def send(path,body,method='POST'):return request(path,method,json.dumps(body).encode())
def person(line):
    name,rest=line.rsplit(' <',1);email,tail=rest.split('> ',1);timestamp,zone=tail.split();offset=(1 if zone[0]=='+' else -1)*(int(zone[1:3])*60+int(zone[3:]))
    date=datetime.datetime.fromtimestamp(int(timestamp),datetime.timezone(datetime.timedelta(minutes=offset))).isoformat()
    return dict(name=name,email=email,date=date)
def import_commit(commit):
    try:
        r=request('git/commits/'+commit)
        if r['sha']==commit:return
    except urllib.error.HTTPError as e:
        if e.code not in (404,422):raise
    raw=git('cat-file','-p',commit).decode();headers,message=raw.split('\n\n',1);d={}
    for line in headers.splitlines():
        k,v=line.split(' ',1);d[k]=v
    parent=d['parent'];base=request('git/commits/'+parent)['tree']['sha'];tree=[]
    names=git('diff','--name-only',parent,commit).decode().splitlines()
    for name in names:
        raw=git('show',commit+':'+name)
        blob=send('git/blobs',dict(content=base64.b64encode(raw).decode(),encoding='base64'))
        expected=git('rev-parse',commit+':'+name).decode().strip();assert blob['sha']==expected
        mode=git('ls-tree',commit,'--',name).decode().split()[0]
        tree.append(dict(path=name,mode=mode,type='blob',sha=expected))
    t=send('git/trees',dict(base_tree=base,tree=tree));assert t['sha']==d['tree']
    r=send('git/commits',dict(message=message,tree=t['sha'],parents=[parent],author=person(d['author']),committer=person(d['committer'])))
    assert r['sha']==commit,(r['sha'],commit)
    print('exact commit objects imported',commit,flush=True)
def main():
    action=sys.argv[1];head=git('rev-parse','HEAD').decode().strip()
    if action=='import':
        remote=request('git/ref/heads/main')['object']['sha'];assert subprocess.run(['git','-C',str(W),'merge-base','--is-ancestor',remote,head]).returncode==0
        for commit in git('rev-list','--reverse',remote+'..'+head).decode().splitlines():import_commit(commit)
        tag=os.environ.get('BT_RELEASE_TAG','web-bun392-a3sr-u1280-20261030-r1');raw=git('cat-file','-p',tag).decode();headers,message=raw.split('\n\n',1);d=dict(line.split(' ',1) for line in headers.splitlines())
        obj=send('git/tags',dict(tag=d['tag'],message=message,object=d['object'],type=d['type'],tagger=person(d['tagger'])))
        assert obj['sha']==git('rev-parse',tag).decode().strip()
        try:r=request('git/ref/tags/'+tag)
        except urllib.error.HTTPError as e:
            if e.code!=404:raise
            r=send('git/refs',dict(ref='refs/tags/'+tag,sha=obj['sha']))
        assert r['object']['sha']==obj['sha']
        (O/'GIT_OBJECT_IMPORT.json').write_text(json.dumps(dict(status='PASS',head=head,tag=tag,tag_object=obj['sha'],exact_local_git_objects=True,force_used=False,transport='official GitHub Git Data API fallback for unavailable smartHTTP',unix=time.time()),indent=2)+'\n');print('exact annotated tag created',tag)
    elif action=='main':
        release=json.loads((O/'RELEASE_RECEIPT.json').read_text());assert release['tag']==os.environ.get('BT_RELEASE_TAG','web-bun392-a3sr-u1280-20261030-r1')
        remote=request('git/ref/heads/main')['object']['sha'];assert subprocess.run(['git','-C',str(W),'merge-base','--is-ancestor',remote,head]).returncode==0
        r=send('git/refs/heads/main',dict(sha=head,force=False),'PATCH');assert r['object']['sha']==head
        (O/'MAIN_PUSH.json').write_text(json.dumps(dict(status='FAST_FORWARD',old=remote,head=head,force=False,transport='official GitHub Git Data API ref update, exact local objects',unix=time.time()),indent=2)+'\n');print('main fast-forward',head)
if __name__=='__main__':main()
