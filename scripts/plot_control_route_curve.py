"""Plot already-completed evidence; never evaluates models or controls runs."""
import argparse
import json
from pathlib import Path


def plot(source,output,round_number=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads(Path(source).read_text())
    if 'rounds' in data:
        if round_number is None or not 1 <= round_number <= len(data['rounds']):
            raise ValueError('bundle input requires a valid --round number')
        data=data['rounds'][round_number-1]
    points=data['checkpoints'];generations=[p['generation'] for p in points]
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    colors=['#0072B2','#D55E00','#009E73','#CC79A7']
    for ax,tasks in zip(axes[0],[['real_drop','carry_far','natural_full2400'],
                                ['native_death_stock','native_sugar_stock','recovery_threat']]):
        for color,task in zip(colors,tasks):
            ax.plot(generations,[p['routes'][task]['delivery_per_game']['numerator'] for p in points],
                    marker='o',label=task,color=color)
        ax.set(ylim=(-.5,16.5),ylabel='Delivery episodes / 16',xlabel='Fixed generation')
        ax.legend(fontsize=9)
    axes[0,0].set_title('Loose pickup / long return / natural full games')
    axes[0,1].set_title('Stock windows and recovery threat')
    ax=axes[1,0]
    for color,tier in zip(colors,['hard','normal','easy']):
        ax.plot(generations,[p['retention']['hunter_js'][tier]['safe_win'] for p in points],
                marker='o',label='JS Hunter '+tier,color=color)
    ax.plot(generations,[p['retention']['tactical_python']['safe_win'] for p in points],
            marker='s',linestyle='--',label='Python Tactical v2',color=colors[3])
    ax.set(title='Combat retention: simulator protocols remain separate',
           ylabel='Safe kills / 64',xlabel='Fixed generation',ylim=(-.5,32))
    ax.legend(fontsize=9)
    ax=axes[1,1]
    for color,event in zip(colors,['own_window_enter','inventory_or_native_steal','capture','loose_pickup']):
        ax.plot(generations,[p['training']['native_full_events_per_million_full_ticks'].get(event,float('nan')) for p in points],
                marker='o',label=event,color=color)
    ax.set(title='JAX training events, normalized by natural-full exposure',
           ylabel='Cumulative events / million full ticks',xlabel='Fixed generation')
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.set_xticks(generations)
        ax.spines[['top','right']].set_visible(False)
    run_label=data['run'].rsplit('/',1)[-1]
    fig.suptitle(run_label+' — all completed declared checkpoints',fontsize=15)
    fig.text(.5,-.02,'Fixed seeds; greedy assist-off. No selected best seed/checkpoint, smoothing, or training gates. '
             'Stock geometry/seat confounding and oracle limitations are documented in the report.',ha='center',fontsize=9)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=170,bbox_inches='tight',metadata={'Software':'JE10 completed-evidence analysis'})
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('output')
    parser.add_argument('--round',type=int,dest='round_number')
    args=parser.parse_args();plot(args.source,args.output,args.round_number)
