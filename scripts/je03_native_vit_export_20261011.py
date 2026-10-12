"""CPU-only fp32 ONNX export for the audited native ViT architecture."""
from pathlib import Path
import pickle
import sys
import numpy as np
import torch
from torch.nn import functional as F


class NativeViT(torch.nn.Module):
    def __init__(self,params):
        super().__init__();self.params=params
        self.depth=len(params['blocks'])
        def convert(x):
            if isinstance(x,dict):return {k:convert(v) for k,v in x.items()}
            if isinstance(x,(tuple,list)):return type(x)(convert(v) for v in x)
            return torch.from_numpy(np.asarray(x,np.float32).copy())
        self.params=convert(params)

    def forward(self,obs,state):
        p=self.params;n=obs.shape[0];c=obs.shape[1];embed=p['tok'][0].shape[1]
        x=F.pad(obs,(0,0,0,2)).reshape(n,c,5,3,5,3).permute(0,2,4,1,3,5).reshape(n,25,c*9)
        x=x@p['tok'][0]+p['tok'][1]+p['pos'][:,:25]
        token=state@p['state_w']+p['state_b']+p['pos'][:,-1]
        x=torch.cat([x,token[:,None]],1)
        for block in p['blocks']:
            z=F.layer_norm(x,(embed,),block['ln1_g'],block['ln1_b'],1e-6)
            q,k,v=[(z@block[key][0]+block[key][1]).reshape(n,26,4,embed//4).permute(0,2,1,3) for key in ('q','k','v')]
            scores=q@k.transpose(-1,-2)/(embed//4)**.5
            attention=(scores.softmax(-1)@v).permute(0,2,1,3).reshape(n,26,embed)
            x=x+attention@block['proj'][0]+block['proj'][1]
            z=F.layer_norm(x,(embed,),block['ln2_g'],block['ln2_b'],1e-6)
            x=x+F.relu(z@block['ff1'][0]+block['ff1'][1])@block['ff2'][0]+block['ff2'][1]
        z=x[:,:25].mean(1)
        move=z@p['heads']['wm'][0]+p['heads']['wm'][1]
        bomb=z@p['heads']['wb'][0]+p['heads']['wb'][1]
        value=z@p['heads']['wv'][0]+p['heads']['wv'][1]
        return move,bomb,(value.softmax(-1)*torch.linspace(-20,20,value.shape[-1])).sum(-1)


def export(source,target):
    torch.set_num_threads(1)
    with Path(source).open('rb') as f:params=pickle.load(f)
    model=NativeViT(params).eval();c=params['tok'][0].shape[0]//9
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():torch.onnx.export(model,(torch.zeros((1,c,13,15)),torch.zeros((1,24))),
        str(target)+'.tmp',input_names=['obs','state'],output_names=['move','bomb','value'],opset_version=17,dynamo=False)
    Path(str(target)+'.tmp').replace(target)


if __name__=='__main__':export(*sys.argv[1:])
