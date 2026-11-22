"""Durable Actor/Critic-only checkpoints with retention scoped to one new run."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def durable_write(path,data):
    with Path(path).open('xb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())


def durable_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path);sync_directory(path.parent)


class CheckpointStore:
    def __init__(self,out,keep=64,budget_gib=4,min_free_gib=8,*,schema='qqt.kill-actor-critic/v1'):
        self.out=Path(out);self.keep=keep
        self.schema=schema
        self.budget=int(budget_gib*1024**3);self.min_free=int(min_free_gib*1024**3)

    def publish(self,index,weights,metadata):
        if metadata.get('schema') != self.schema:
            raise ValueError('Checkpoint schema does not match this store')
        # Only the two explicitly named model trees can enter the payload.
        payload={b:pickle.dumps(weights[b],protocol=5) for b in ('actor','critic')}
        size=sum(map(len,payload.values()))+16384
        if size>self.budget:raise OSError('One checkpoint exceeds the configured storage budget')
        if shutil.disk_usage(self.out).free<size+self.min_free:
            raise OSError('Checkpoint would violate the free-space reserve; previous LATEST is preserved')
        name=f'u{index:06d}';target=self.out/name;staging=self.out/('.'+name+'.tmp')
        if target.exists() or staging.exists():raise FileExistsError(target)
        staging.mkdir()
        try:
            for branch,data in payload.items():durable_write(staging/(branch+'.pt'),data)
            meta=dict(metadata,update=index,optimizer_saved=False,precise_resume=False,
                **{b+'_sha256':hashlib.sha256(data).hexdigest() for b,data in payload.items()})
            durable_json(staging/'CHECKPOINT.json',meta)
            staging.replace(target);sync_directory(self.out)
            durable_json(self.out/'LATEST.json',dict(checkpoint=name,**meta))
        except BaseException:
            if staging.exists():shutil.rmtree(staging)
            raise
        deleted=[];entries=[]
        # Never traverse symlinks, historical runs or arbitrary directories.
        for path in self.out.iterdir():
            if path.is_symlink() or not re.fullmatch(r'u\d{6,}',path.name) or not path.is_dir():continue
            marker=path/'CHECKPOINT.json'
            if not marker.is_file():continue
            record=json.loads(marker.read_text())
            if record.get('schema')!=self.schema:continue
            nbytes=sum(p.stat().st_size for p in path.iterdir() if p.is_file())
            entries.append((int(path.name[1:]),path,nbytes))
        entries.sort();total=sum(e[2] for e in entries)
        for entry in entries[:]:
            if len(entries)<=self.keep and total<=self.budget:break
            # Retain the initial model and the newest durable checkpoint.
            step,path,nbytes=entry
            if step in (0,index):continue
            shutil.rmtree(path);entries.remove(entry);total-=nbytes;deleted.append(path.name)
        sync_directory(self.out)
        receipt=dict(latest=name,checkpoints=len(entries),checkpoint_bytes=total,
            max_checkpoints=self.keep,budget_bytes=self.budget,min_free_bytes=self.min_free,
            pruned=deleted,optimizer_saved=False,initial_checkpoint_preserved=True)
        durable_json(self.out/'STORAGE.json',receipt)
        return receipt
