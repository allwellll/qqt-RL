'use strict';
(function(root, factory) {
  const api = factory(typeof module === 'object' && module.exports ? require('./sim') : root.QQT);
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.QQTReviewView = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function(Q) {
  const REWARDS = Object.freeze({
    route_potential_delta: ['路线势能变化', 'γΦ(s_next)−Φ(s)，人工路线诊断量，不是 Critic 优势'],
    route_potential_positive: ['路线正势能差', '路线势能差的正值统计，不额外叠加奖励'],
    route_potential_negative: ['路线负势能差', '路线势能差的负值统计，不额外叠加奖励'],
    route_shaped_reward: ['路线塑形奖励', 'λ×路线势能差，实际 reward ledger 中的人工辅助奖励'],
    route_topology_gain: ['路线拓扑增益', '真实稳定开墙和路长改善诊断，不再叠加第二份奖励'],
    route_phase_transition: ['路线阶段转换', '目标阶段转换诊断'],
    route_invalidated: ['路线目标失效', '目标失效与结算诊断'],
    route_terminal_correction: ['路线终态修正', '终态势能归零的边界修正诊断'],
    control_entry_reward: ['首次控制收益', '同版本JE10账本对独占因果归因的首次糖泡控制付分'],
    control_residual_reward: ['控制终结残余收益', '已有控制后真实主动终结的残余收益'],
    control_clawback: ['控制收益追回', '死亡时追回尚未结算的控制余额'],
    lesson_task: ['任务收益', '课程目标与包子任务的基础收益'],
    global_safety: ['全局安全', '自炸、可避免死亡等全局安全奖惩'],
    settled_kill: ['结算击杀', '危险消退后结算的有效击杀收益'],
    settled_shaping: ['结算战斗塑形', '危险消退后结算的战斗辅助收益'],
    combat_clawback: ['战斗收益追回', '撤销未满足结算条件的战斗收益'],
    blocked_route: ['阻断路线', '阻断己方运输路线的惩罚'],
    transport_shape: ['运输过程塑形', '运输窗口中的目标接近与路径收益'],
    transport_clawback: ['运输收益追回', '撤销未兑现的运输过程收益'],
    transport_capture: ['运输交付', '把敌方包子运回己笼的收益'],
    drop_bonus: ['制造掉包', '使敌方持包者掉包的收益'],
    drop_clawback: ['掉包收益追回', '撤销未兑现的制造掉包收益'],
    drop_approach_net: ['接近掉包净收益', '朝掉落包子接近的净收益'],
    drop_approach_clawback: ['接近掉包收益追回', '撤销未兑现的接近掉包收益'],
    carry_home_net: ['持包返程净收益', '持包向己方基地前进的净收益'],
    carry_home_clawback: ['持包返程收益追回', '撤销未兑现的持包返程收益'],
    bridge_reward: ['运输桥接收益', '连接战斗窗口、取包与返程的桥接收益'],
    bridge_clawback: ['桥接收益追回', '撤销未兑现的运输桥接收益'],
    item_positive: ['道具正收益', '有效使用运输与战斗道具的收益'],
    item_clawback: ['道具收益追回', '撤销未兑现的道具收益'],
  });
  function reward(key) {
    const known = Object.hasOwn(REWARDS, key);
    return { key, known, name: known ? REWARDS[key][0] : `未翻译（${key}）`,
      help: known ? `${REWARDS[key][1]} · ${key}` : `未翻译（${key}）` };
  }
  function componentNames(doc) {
    return [...new Set([...(doc.meta.reward_components || []),
      ...doc.rewards.flatMap(row => [...Object.keys(row[1]), ...Object.keys(row[2])])])];
  }
  const DISCONTINUITIES = new Set(['death','revive','control','control_end','drop','pickup',
    'native_transfer','stock_theft','delivery','use_item','self_bomb']);
  const MAX_DISTANCE = 0.6;
  function interpolationReason(before, after, pid, events = [], intent) {
    const production=typeof before._tryMove==='function'?before:Q.Sim.prototype;
    if (!after || after.t !== before.t+1)return 'missing_adjacent_tick';
    if(before.done||after.done)return 'terminal';
    if(!before.alive[pid]||!after.alive[pid])return 'death_or_respawn';
    if(before.trapped[pid]||after.trapped[pid])return 'control';
    for (const key of ['bunCarried','heldItem','hp','movementStatus','slideDir','spdG']) {
      if (before[key]?.[pid] !== after[key]?.[pid]) return 'state_change:'+key;
    }
    for (const key of ['spawnProtection','invuln','bunRespawn']) {
      if (Boolean(before[key]?.[pid]) !== Boolean(after[key]?.[pid])) return 'threshold:'+key;
    }
    const boundary=events.find(e => e.tick===after.t && e.player===pid && DISCONTINUITIES.has(e.kind));
    if(boundary)return 'event:'+boundary.kind;
    const y=before.pos[pid*2],x=before.pos[pid*2+1],ny=after.pos[pid*2],nx=after.pos[pid*2+1];
    const origin=Math.floor(y)*15+Math.floor(x);
    if(before.wall[origin]||before.brick[origin])return 'solid_origin';
    const placing=events.some(e=>e.tick===after.t&&e.player===pid&&e.kind==='bomb');
    const ownPlacement=placing&&!before.fuse[origin]&&after.fuse[origin]>0&&after.owner?.[origin]===pid;
    if(placing&&!ownPlacement)return 'unverified_bomb';
    const distance=Math.abs(ny-y)+Math.abs(nx-x);
    if (!Number.isFinite(distance)||distance>MAX_DISTANCE)return 'distance_threshold';
    if(distance<1e-8)return 'stationary';
    if(Math.abs(ny-y)>1e-8 && Math.abs(nx-x)>1e-8)return 'collision_alignment';
    const direction=ny<y?0:ny>y?1:nx<x?2:3;
    // Side steering, collision clamps and alignment corrections snap at the boundary.
    if(intent!==undefined&&direction!==intent)return 'collision_side_steering';
    if(intent!==undefined&&Math.abs(distance-Q.CFG.stepLen*before.spdG[pid]*production.playerMoveScale.call(before,pid))>1e-6)return 'collision_clamp';
    const blocked=new Uint8Array(195);
    for(let i=0;i<195;i++)blocked[i]=Boolean(before.wall[i]||before.brick[i]||before.fuse[i]>0||after.wall[i]||after.brick[i]||after.fuse[i]>0);
    // Use production collision resolution, including its start-cell exemption.
    for(let n=1;n<=8;n++){
      const f=n/8,actual=production._tryMove.call(before,y,x,direction,blocked,distance*f);
      if(Math.abs(actual[0]-(y+(ny-y)*f))>1e-7||Math.abs(actual[1]-(x+(nx-x)*f))>1e-7)return 'collision_sweep';
    }
    const radius=.36;
    for(let r=Math.floor(Math.min(y,ny)-radius);r<=Math.floor(Math.max(y,ny)+radius);r++) {
      for(let c=Math.floor(Math.min(x,nx)-radius);c<=Math.floor(Math.max(x,nx)+radius);c++) {
        if(r<0||r>=13||c<0||c>=15)continue;
        const i=r*15+c;
        for(const key of ['wall','brick','pushable','crate','fuse','fieldItem']) {
          if(Boolean(before[key]?.[i])!==Boolean(after[key]?.[i])){
            // A new own bubble stays discrete. Production permits leaving its start cell;
            // its creation alone does not invalidate an independently swept straight move.
            if(key==='fuse'&&i===origin&&ownPlacement)continue;
            return 'topology_change:'+key;
          }
        }
      }
    }
    return null;
  }
  const canInterpolate=(before,after,pid,events=[],intent)=>interpolationReason(before,after,pid,events,intent)===null;
  function motion(before, after, fraction, enabled, events = [], intents) {
    const curPos=Array.from(before.pos),smooth=[],snapReasons=[];
    for(let p=0;p<before.alive.length;p++) {
      snapReasons[p]=interpolationReason(before,after,p,events,intents?.[p]);
      smooth[p]=Boolean(enabled && snapReasons[p]===null);
      if(smooth[p]){curPos[p*2]=after.pos[p*2];curPos[p*2+1]=after.pos[p*2+1];}
    }
    const alpha=enabled?Math.max(0,Math.min(1,fraction)):0;
    const visualPos=Array.from(before.pos,(v,i)=>v+(curPos[i]-v)*alpha);
    return {humanPid:-1,tickMs:100,lastTickT:before.t*100,prevPos:Array.from(before.pos),curPos,visualPos,
      fraction:enabled?Math.max(0,Math.min(1,fraction)):0,smooth,snapReasons,sourceTick:before.t,nextTick:after?.t??null,
      characters:['maomao','maomao'],playerPalettes:[null,'blue'],playerLabels:['模型 P0','对手 P1'],hideResult:true};
  }
  return { REWARDS, reward, componentNames, MAX_DISTANCE, interpolationReason, canInterpolate, motion };
});
