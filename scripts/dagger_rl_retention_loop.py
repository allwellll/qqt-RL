"""Inactive bounded loss-schedule loop preparation, with no dispatch side effects.

Reuse the previously frozen two categorical learner collectors and telemetry.
An owner must supply a new, separately authorized namespace before execution.
"""
import inspect

from scripts.dagger_rl_retention_schedule import branch_for_update, make_update
from scripts.dagger_rl_item_opportunity_telemetry import summarize
from scripts.carrier_mechanism_train import preserve_critic


def formal_function(template, *, bind, base, root, read, authorize, notify, arm, seed):
    source = inspect.getsource(template.formal)

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('frozen training-loop anchor changed: '+old)
        source = source.replace(old, new)

    replace("for seed in pr['paired_seeds']:", 'for seed in [requested_seed]:')
    replace("for arm in pr['arms']:", 'for arm in [requested_arm]:')
    replace("update=C.updater('anchor',opt);aux_collect=aux_collectors[arm]",
            "updates={'joint_anchor':C.updater('anchor',opt),"
            "'dagger_anchor':make_schedule_update(C.objective,opt,'dagger_anchor',preserve_critic),"
            "'ppo_anchor':make_schedule_update(C.objective,opt,'ppo_anchor',preserve_critic)};"
            'aux_collect=aux_collectors[arm]')
    replace("authorize();before=one(p);mode='anchor'",
            "authorize();before=one(p);branch=branch_for_update(arm,u);"
            "mode='dagger' if branch=='dagger_anchor' else 'ppo' if branch=='ppo_anchor' else 'anchor';"
            'update=updates[branch]')
    replace("row['dagger_coefficient']=pr['dagger_coefficient_by_arm'][arm]",
            "row['dagger_coefficient']=0. if mode=='ppo' else .2\n"
            "    row['loss_branch']=branch\n"
            "    row['adopted_fresh_PPO_environment_steps_this_update']=0 if mode=='dagger' else 8192\n"
            "    row['supervised_loss_optimizer_steps_this_update']=0 if mode=='ppo' else 32\n"
            "    row['critic_parameter_updates_this_update']=0 if mode=='dagger' else 32\n"
            "    row['item_opportunity_denominators']=passive_item_summary(diag[1],diag[2],diag[6])\n"
            "    row['auxiliary_item_opportunity_denominators']=passive_item_summary(aux_diag[1],aux_diag[2],aux_diag[6])")
    replace(" atomic_json(B/'TRAINING_COMPLETE.json',dict(passed=True,runs=records,environment_steps=3145728,optimizer_steps=6144,old_continuous_resume=False))",
            " assert len(records)==1 and records[0]['update']==32")
    source = source.replace('R12', 'R16')
    source = source.replace('中段邻点完整保存（本轮无阶段切换）',
                            '固定16/17阶段边界完整保存（仅顺序臂切换）')
    scope = dict(vars(template), B=base, R=root, read=read, authorize=authorize,
                 notify=notify, bind=bind, requested_arm=arm, requested_seed=seed,
                 branch_for_update=branch_for_update, make_schedule_update=make_update,
                 preserve_critic=preserve_critic, passive_item_summary=summarize)
    exec(compile(source, '<bounded-retention-schedule-loop>', 'exec'), scope)
    return scope['formal']
