"""JE09 fresh-state bridge continuation contracts."""
import copy
import os
import json
from pathlib import Path

ARM = "bun392_bridge_pfsp8"
SHAPE = [8, 1024, 256, 4096]
SCHEMA = "je09-eight-device-drop-active-shortest-v4"
NAMESPACE = os.environ.get("JE09_NAMESPACE", "runs/je09_bun_20261015/stage_b_window_pfsp8_v6")
SOURCE_NAMESPACE = "runs/je08_bun_20261014/h200_dp8_continuation_g1120_v1"
SOURCE_CHECKPOINT = SOURCE_NAMESPACE + "/formal/bun392_topology_potential_pfsp8/gen002048"
STAGE_D = "stage_d_" in NAMESPACE
STAGE_C = "stage_c_" in NAMESPACE or STAGE_D
STAGE_B = "stage_b_" in NAMESPACE or STAGE_C
if STAGE_B:
    SCHEMA = "je09-eight-device-stage-b-window-v6"
    SOURCE_NAMESPACE = "runs/je09_bun_20261015/drop_active_local_safe_pfsp8_v5"
    SOURCE_CHECKPOINT = SOURCE_NAMESPACE + "/formal/bun392_bridge_pfsp8/gen000008"
if STAGE_C:
    SCHEMA = "je09-eight-device-stage-c-return-set-v8"
    SOURCE_NAMESPACE = "runs/je09_bun_20261015/stage_b_window_pfsp8_v7"
    SOURCE_CHECKPOINT = SOURCE_NAMESPACE + "/formal/bun392_bridge_pfsp8/gen000008"
if STAGE_D:
    SCHEMA="je09-eight-device-stage-d-distance-set-v9"
    SOURCE_NAMESPACE="runs/je09_bun_20261015/stage_c_return_set_pfsp8_v8"
    SOURCE_CHECKPOINT=SOURCE_NAMESPACE+"/formal/bun392_bridge_pfsp8/gen000008"
SOURCE_EVALUATION = ("authority_je09_v1" if STAGE_B else "authority_je08_v1") + "/EVALUATION_COMPLETE.json"
REPLICATED = ("params", "adam", "ema", "funnel", "pool_counts", "course", "pfsp",
              "assist_level", "assist_proof_hashes")


def continuation_plan(source):
    plan = copy.deepcopy(source)
    plan.update(
        experiment_id="JE09-DROP-ACTIVE-SHORTEST-PFSP8-20261015-v4",
        arms=[ARM], devices={ARM: "0,1,2,3,4,5,6,7"}, arms_shape=SHAPE.copy(),
        envs_per_device=1024, gradient_accumulation=1, state_schema=SCHEMA,
        physical_global_microbatch=32768, effective_global_actor_batch=16384,
        seed=2026101541,ring_interval=8,combat_fraction=.40,
        effective_global_critic_batch=16384, generations=1024,
        checkpoints=[0, 1, 8, 32, 64, 128, 256, 512, 768, 1024],
        continuation=dict(source_checkpoint=SOURCE_CHECKPOINT, source_generation=2048,
            optimizer_reset=True, rng_reset=True, env_reset=True,
            bridge_manifest="je09.real-drop-bridge/v1",
            seed_rule="train/diagnostic/evaluation namespaces are fixed and holdout is disjoint",
            batch_accounting="Actor and Critic each consume half of physical microbatch"),
        campaign=dict(stage="bridge", assist_off_required=True,
                      total_budget_generations=32,total_budget_fresh_ticks=67108864,
                      max_rounds=2,short_round_generations=16,
                      round=1,completed_formal_ticks_before_round=0,
                      change='attributed empty-handed legal shortest-path movement auxiliary supervision; preserve R3 reward and pool',
                      mixture={"combat_parent": .40, "bridge": .35, "other_prior": .25},
                      promotion="bridge first_pickup and capture plus Hunter/Tactical retention"),
    )
    plan['chain_scene'].update(weight=.35,trap_ticks=1,drop_distance=[2,8],
                               fixture='predeclared learner-attributed trap; native env.step drop; not kill evidence')
    plan['evaluation']['namespace']='authority_je09_v1'
    plan['reward'].update(profile='je09_real_drop_bridge_fixturekill_v3',
        composition='native pickup8/capture24 once + JE08 signed topology + gated post-pickup milestones; no predeclared scene9 t0 kill payment')
    plan['extension_rule']='evaluate each short round; change smallest evidenced intervention; at most campaign total budget'
    scan=json.loads(Path("runs/je09_bun_20261015/drop_active_shortest_pfsp8_v4/validation/EXPERT_SCAN.json").read_text())
    plan["drop_expert"]=dict(coefficient=scan["selected"]["strength"],temperature=scan["selected"]["temperature"],action_override=False)
    if NAMESPACE.endswith("_v5"):
        override=json.loads((Path(NAMESPACE)/"ROUND_DECLARATION.json").read_text())
        plan["drop_expert"].update(override["drop_expert"])
        plan["seed"]=2026101551
        plan["campaign"].update(round=2,completed_formal_ticks_before_round=33554432,change=override["change"])
        plan["experiment_id"]="JE09-DROP-ACTIVE-LOCAL-SAFE-PFSP8-20261015-v5"
    if STAGE_B:
        declaration=json.loads((Path(NAMESPACE)/"ROUND_DECLARATION.json").read_text())
        plan["experiment_id"]=declaration["experiment_id"]
        plan["seed"]=declaration["seed"]
        plan["drop_expert"]=dict(coefficient=.05,temperature=.5,hazard_mode="local_safe",action_override=False)
        plan["stage_b_reward"]=declaration["stage_b_reward"]
        plan["combat_fraction"]=.60
        plan["bridge_fraction"]=.25
        plan["campaign"]["mixture"]={"combat_parent":.60,"bridge":.25,"other_prior":.15}
        plan["campaign"]["combat_actual_valid_minimum"]=.40
        plan["continuation"].update(source_checkpoint=SOURCE_CHECKPOINT,source_generation=8)
        plan["evaluation"]["namespace"]="authority_je09_v1"
        plan["campaign"].update(stage="B",total_budget_generations=16,total_budget_fresh_ticks=33554432,
            short_round_generations=8,round=declaration["round"],change=declaration["change"],
            completed_formal_ticks_before_round=(declaration["round"]-1)*16777216)
    if STAGE_C:
        plan['return_expert']=declaration['return_expert']
        plan['ring_generation_offset']=8
        assert plan['return_expert']['coefficient'] is not None
        plan['campaign'].update(stage='C',total_budget_generations=8,total_budget_fresh_ticks=16777216,
            completed_formal_ticks_before_round=0,max_rounds=1)
        plan['evaluation']['stage_c_seeds']=declaration['seeds']
    if STAGE_D:
        plan['drop_expert']=declaration['drop_expert']
        plan['distance_curriculum']=declaration['curriculum']
        plan['distance_buckets']=declaration['distance_buckets']
        plan['ring_generation_offset']=16
        plan['campaign'].update(stage='D',total_budget_generations=16,total_budget_fresh_ticks=33554432)
    return plan


def validate_shape(plan):
    assert plan["arms_shape"] == SHAPE
    assert plan["global_envs"] == 8192
    assert plan["fresh_ticks_per_generation"] == 2097152
    assert plan["updates_per_generation"] == 128
    return True
