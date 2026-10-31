# R3-sp 自对弈递进（20261031）

原R3-a/b冻结u4002作对照；新arm首阶段55%自对弈、45% bot，固定640更新。奖励、规则、网络、LR和batch不改。
父点 `runs/r3_items_route_20261030/formal/r3_a_c000000/u003040`；policy SHA `80ae9005b1563b1ba7a1cbc07ed9044ea051f466fd47b509b2759e7c174f36d2`。
冻结协议与原始证据：`runs/r3_selfplay_ramp_20261031/PROTOCOL.json`、`PARENT_GATE.json`、`REGISTRATION.json`。
current=每更新的当前Actor；ring=4份历史Actor、每16更新替换一份；fixed=3份冻结历史强点，按环境编号静态均分。没有PFSP采样。
当前run `r3_sp55_c000000`；最新真实更新 3520；数值finite=True。
首正式更新以 `FIRST_FORMAL_UPDATE.json` 为准，launch/进程存在不算训练；全部原子保存含Actor/Critic/Adam/env/aux/RNG/ring。

|对手池|实际环境tick|实际新局|
|---|---:|---:|
|rule3|4976640|7796|
|jax_hunter|18370560|30939|
|jax_hunter_hard|30597120|48871|
|current|31395840|38810|
|ring|27279360|34989|
|fixed|13209600|14304|

自对弈监控：同为20×262144 train步窗口，按P0/P1及current/ring/fixed分母；仅自然完整局，Bank、清场场景、终局账本尾段不混入。
固定对手Elo是自然训练局 score=胜+半平 对应的分差，不冒充独立保留集Elo联赛或统计显著性。
循环计数为“观察到击杀己方来源持包者→同源本家库存真实增加”的来源级序列，未保存唯一包ID，不能证明行为意图或同一枚包。
被动局=双方都没有进入过敌笼，不等同已证明合谋；按用户规则阻止进一步加比例，当前健康55%预算继续。

```json
{
  "status": "ROLLBACK",
  "seats": {
    "P0": {
      "games": 1156.0,
      "parent_games": 841.0,
      "draw_rate": 0.054498269896193774,
      "parent_draw_rate": 0.013079667063020214,
      "passive_rate": 0.015570934256055362,
      "parent_passive_rate": 0.0047562425683709865,
      "deliveries": 527.0,
      "parent_deliveries": 456.0,
      "equal_environment_step_budget": true,
      "flags": {
        "draw_15pp": false,
        "passive_increase": true,
        "delivery_halved": false
      },
      "own_zero_bomb": 0.0,
      "both_zero_bomb": 0.0,
      "custody_loops": 73.0,
      "unresolved": 0.0,
      "by_pool": {
        "current": {
          "active_deliveries": 242.0,
          "both_zero_bomb": 0.0,
          "custody_loops": 30.0,
          "draws": 23.0,
          "episodes": 530.0,
          "losses": 272.0,
          "own_zero_bomb": 0.0,
          "passive": 3.0,
          "unresolved": 0.0,
          "wins": 235.0
        },
        "ring": {
          "active_deliveries": 182.0,
          "both_zero_bomb": 0.0,
          "custody_loops": 27.0,
          "draws": 15.0,
          "episodes": 442.0,
          "losses": 250.0,
          "own_zero_bomb": 0.0,
          "passive": 0.0,
          "unresolved": 0.0,
          "wins": 177.0
        },
        "fixed": {
          "active_deliveries": 103.0,
          "both_zero_bomb": 0.0,
          "custody_loops": 16.0,
          "draws": 25.0,
          "episodes": 184.0,
          "losses": 53.0,
          "own_zero_bomb": 0.0,
          "passive": 15.0,
          "unresolved": 0.0,
          "wins": 106.0
        }
      }
    },
    "P1": {
      "games": 1060.0,
      "parent_games": 847.0,
      "draw_rate": 0.04716981132075472,
      "parent_draw_rate": 0.0070838252656434475,
      "passive_rate": 0.011320754716981131,
      "parent_passive_rate": 0.0,
      "deliveries": 561.0,
      "parent_deliveries": 360.0,
      "equal_environment_step_budget": true,
      "flags": {
        "draw_15pp": false,
        "passive_increase": true,
        "delivery_halved": false
      },
      "own_zero_bomb": 0.0,
      "both_zero_bomb": 0.0,
      "custody_loops": 74.0,
      "unresolved": 0.0,
      "by_pool": {
        "current": {
          "active_deliveries": 271.0,
          "both_zero_bomb": 0.0,
          "custody_loops": 35.0,
          "draws": 24.0,
          "episodes": 484.0,
          "losses": 200.0,
          "own_zero_bomb": 0.0,
          "passive": 1.0,
          "unresolved": 0.0,
          "wins": 260.0
        },
        "ring": {
          "active_deliveries": 229.0,
          "both_zero_bomb": 0.0,
          "custody_loops": 19.0,
          "draws": 12.0,
          "episodes": 418.0,
          "losses": 184.0,
          "own_zero_bomb": 0.0,
          "passive": 1.0,
          "unresolved": 0.0,
          "wins": 222.0
        },
        "fixed": {
          "active_deliveries": 61.0,
          "both_zero_bomb": 0.0,
          "custody_loops": 20.0,
          "draws": 14.0,
          "episodes": 158.0,
          "losses": 82.0,
          "own_zero_bomb": 0.0,
          "passive": 10.0,
          "unresolved": 0.0,
          "wins": 62.0
        }
      }
    }
  },
  "fixed": {
    "0": {
      "P0": {
        "games": 34.0,
        "wins": 9.0,
        "draws": 13.0,
        "score": 0.45588235294117646,
        "unresolved": 0.0,
        "resolved_games": 34.0,
        "elo_difference": -30.73601209308895
      },
      "P1": {
        "games": 73.0,
        "wins": 18.0,
        "draws": 9.0,
        "score": 0.3082191780821918,
        "unresolved": 0.0,
        "resolved_games": 73.0,
        "elo_difference": -140.44354400291957
      }
    },
    "1": {
      "P0": {
        "games": 45.0,
        "wins": 21.0,
        "draws": 9.0,
        "score": 0.5666666666666667,
        "unresolved": 0.0,
        "resolved_games": 45.0,
        "elo_difference": 46.602227628574866
      },
      "P1": {
        "games": 40.0,
        "wins": 14.0,
        "draws": 2.0,
        "score": 0.375,
        "unresolved": 0.0,
        "resolved_games": 40.0,
        "elo_difference": -88.73949984654256
      }
    },
    "2": {
      "P0": {
        "games": 105.0,
        "wins": 76.0,
        "draws": 3.0,
        "score": 0.7380952380952381,
        "unresolved": 0.0,
        "resolved_games": 105.0,
        "elo_difference": 179.9876034704191
      },
      "P1": {
        "games": 45.0,
        "wins": 30.0,
        "draws": 3.0,
        "score": 0.7,
        "unresolved": 0.0,
        "resolved_games": 45.0,
        "elo_difference": 147.19071411783773
      }
    }
  },
  "scope": "train natural full-episode categorical; fixed actor immutable per environment; not independent dev/holdout",
  "excluded": "genuine Bank, clear arena, ledger tails",
  "monitor_denominator_minimum": 64,
  "unix": 1793457525.4748662,
  "last_update": 3520
}
```

|已完成bot holdout|对手/席位|局数|W/D/L|胜负差|父点差|
|---|---|---:|---|---:|---:|
|parent|hunter_hard/P0|64|52/5/7|45|父点|
|parent|hunter_hard/P1|64|52/9/3|49|父点|
|parent|coop_hard/P0|64|59/3/2|57|父点|
|parent|coop_hard/P1|64|51/6/7|44|父点|
|r3_sp55_c000000_u003040|hunter_hard/P0|64|52/5/7|45|0|
|r3_sp55_c000000_u003040|hunter_hard/P1|64|52/9/3|49|0|
|r3_sp55_c000000_u003040|coop_hard/P0|64|59/3/2|57|0|
|r3_sp55_c000000_u003040|coop_hard/P1|64|51/6/7|44|0|
|r3_sp55_c000000_u003080|hunter_hard/P0|64|49/10/5|44|-1|
|r3_sp55_c000000_u003080|hunter_hard/P1|64|50/7/7|43|-6|
|r3_sp55_c000000_u003080|coop_hard/P0|64|56/3/5|51|-6|
|r3_sp55_c000000_u003080|coop_hard/P1|64|54/7/3|51|7|
|r3_sp55_c000000_u003120|hunter_hard/P0|64|48/8/8|40|-5|
|r3_sp55_c000000_u003120|hunter_hard/P1|64|51/8/5|46|-3|
|r3_sp55_c000000_u003120|coop_hard/P0|64|54/5/5|49|-8|
|r3_sp55_c000000_u003120|coop_hard/P1|64|59/2/3|56|12|
|r3_sp55_c000000_u003160|hunter_hard/P0|64|48/14/2|46|1|
|r3_sp55_c000000_u003160|hunter_hard/P1|64|44/17/3|41|-8|
|r3_sp55_c000000_u003160|coop_hard/P0|64|52/7/5|47|-10|
|r3_sp55_c000000_u003160|coop_hard/P1|64|52/10/2|50|6|
|r3_sp55_c000000_u003200|hunter_hard/P0|64|51/10/3|48|3|
|r3_sp55_c000000_u003200|hunter_hard/P1|64|39/15/10|29|-20|
|r3_sp55_c000000_u003200|coop_hard/P0|64|43/13/8|35|-22|
|r3_sp55_c000000_u003200|coop_hard/P1|64|45/3/16|29|-15|
|r3_sp55_c000000_u003240|hunter_hard/P0|64|55/4/5|50|5|
|r3_sp55_c000000_u003240|hunter_hard/P1|64|54/9/1|53|4|
|r3_sp55_c000000_u003240|coop_hard/P0|64|51/5/8|43|-14|
|r3_sp55_c000000_u003240|coop_hard/P1|64|61/3/0|61|17|
|r3_sp55_c000000_u003280|hunter_hard/P0|64|57/5/2|55|10|
|r3_sp55_c000000_u003280|hunter_hard/P1|64|49/11/4|45|-4|
|r3_sp55_c000000_u003280|coop_hard/P0|64|57/2/5|52|-5|
|r3_sp55_c000000_u003280|coop_hard/P1|64|59/3/2|57|13|

缺失数据：尚无COMPLETE的评估不记为通过；每个新点与父点同128局/条件、各席64，生产JS greedy、老师关闭、2400tick。
fixed胜率/Elo曲线各40更新窗口保存在 `runs/r3_selfplay_ramp_20261031/monitor_points/`；逐更新原始exposure可完整重算，各席各pool的零泡/循环/交付/未知终局不混分母。
不可用：没有PFSP池占比；没有唯一包ID级循环归因；训练on-policy categorical与生产JS greedy分开。
护栏：任一hard/coop席位比父点WL低超过6局，安全原子停新arm，原R3-a/b保持冻结；平局+15pp/被动率升/交付减半只冻结更高比例。
至少640更新后、bot不降且fixed成绩上升，且无被动化阻塞，才可冻结下一阶段70%（bot≥30%）；未达标不悄改比例。
