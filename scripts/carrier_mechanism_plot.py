"""Standalone figures from immutable diagnostic measurements."""
import pickle
import numpy as np
from scripts.carrier_mechanism_run import B,R,read


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=read(B/'PROTOCOL.json');ab=read(B/'AB_COMPLETE.json');c=read(B/'C_NATIVE_BANK_COMPLETE.json');comparison=read(B/'FINAL_COMPARISON.json')
    baseline=read(B/'C/no_update/u000/native_holdout.json');fig,axes=plt.subplots(3,2,figsize=(12,13),layout='constrained')
    for name in p['AB']['models']:
        a=pickle.load((B/'AB'/name/'natural.pkl').open('rb'))['arrays'];y=a['returns'].reshape(-1,8).mean(-1);v=a['v'].reshape(-1,8).mean(-1)
        axes[0,0].scatter(y,v,s=15,alpha=.4,label=name)
    axes[0,0].axline((0,0),slope=1,color='black',linewidth=1);axes[0,0].set(xlabel='Independent terminal MC (8 repeats / state)',ylabel='Critic V',title='A: critic calibration');axes[0,0].legend(fontsize=8)
    labels=[];k=0
    for name,r in ab['models'].items():
        for contrast,q in r['Q'].items():
            lo,hi=q['source_episode_cluster_CI'];m=q['mean'];axes[0,1].errorbar(m,k,xerr=[[max(0,m-lo)],[max(0,hi-m)]],fmt='o',capsize=3);labels.append(name+' '+contrast.replace('forward_minus_','F-'));k+=1
    axes[0,1].set_yticks(range(k),labels,fontsize=8);axes[0,1].axvline(0,color='black',linewidth=1);axes[0,1].set(xlabel='Terminal Q contrast, source-episode 95% CI',title='B: paired action subset (not strict advantage)')
    for color,arm in zip(['tab:blue','tab:orange','tab:green'],p['C']['arms']):
        curves=c['dev_curves'][arm];u=p['C']['save_updates'];a=np.array([[curves[str(seed)][str(t)]['direct_custody_delivery'] for t in u] for seed in p['C']['paired_seeds']])
        for row in a:axes[1,0].plot(u,row,alpha=.2,color=color,linewidth=.8)
        axes[1,0].plot(u,a.mean(0),label=arm,color=color,marker='o');axes[1,0].fill_between(u,a.min(0),a.max(0),color=color,alpha=.1)
        final=c['summary']['holdout']['arms'][arm];vals=np.array(final['direct_custody_delivery']);xx=['PPO-only','DAgger-only','joint'].index(arm);offset=np.linspace(-.1,.1,len(vals))
        axes[1,1].scatter(np.full(len(vals),xx)+offset,vals,color=color)
        natural=comparison['C_retention'][arm]['natural'];counts=[row['episode_counts']['delivery']/row['games'] for row in natural]
        axes[2,0].scatter(np.full(5,xx)+offset,counts,color=color)
        axes[2,1].scatter(np.full(5,xx)+offset,final['death'],color=color)
    axes[1,0].set(xlabel='Fixed outer updates',ylabel='Direct initial-custody delivery',ylim=(0,1),title='C: dev, 5 paired seeds (range band)');axes[1,0].legend(fontsize=8)
    for axis in [axes[1,1],axes[2,0],axes[2,1]]:axis.set_xticks(range(3),p['C']['arms'])
    axes[1,1].axhline(baseline['direct_custody_delivery'],color='black',linestyle='--',label='no update');axes[1,1].set(ylabel='Direct initial-custody delivery',ylim=(0,1),title='C: final holdout, 32 sources x 8 tails');axes[1,1].legend(fontsize=8)
    control=next(iter(comparison['C_retention'].values()))['natural_control'];axes[2,0].axhline(control['episode_counts']['delivery']/control['games'],color='black',linestyle='--',label='shared no update')
    axes[2,0].set(ylabel='Delivery episodes / 64',ylim=(0,.5),title='Production JS natural full games, 5 seeds');axes[2,0].legend(fontsize=8)
    axes[2,1].axhline(baseline['death'],color='black',linestyle='--',label='no update');axes[2,1].set(ylabel='Death probability in finite conditional task',ylim=(0,.2),title='Native holdout safety, 5 seeds');axes[2,1].legend(fontsize=8)
    path=R/'reports/carrier_mechanism_diagnostic_20261023.png';fig.savefig(path,dpi=180);plt.close(fig)
