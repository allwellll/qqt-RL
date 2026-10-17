'use strict';
(function(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.QQTReviewView = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {
  const REWARDS = Object.freeze({
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
  function canInterpolate(before, after, pid, events = []) {
    if (!after || after.t !== before.t+1 || before.done || after.done ||
        !before.alive[pid] || !after.alive[pid] || before.trapped[pid] || after.trapped[pid]) return false;
    for (const key of ['bunCarried','heldItem','hp','movementStatus','slideDir','spdG']) {
      if (before[key]?.[pid] !== after[key]?.[pid]) return false;
    }
    for (const key of ['spawnProtection','invuln','bunRespawn']) {
      if (Boolean(before[key]?.[pid]) !== Boolean(after[key]?.[pid])) return false;
    }
    if (events.some(e => e.tick===after.t && e.player===pid && DISCONTINUITIES.has(e.kind))) return false;
    const y=before.pos[pid*2],x=before.pos[pid*2+1],ny=after.pos[pid*2],nx=after.pos[pid*2+1];
    const distance=Math.abs(ny-y)+Math.abs(nx-x);
    if (!Number.isFinite(distance) || distance>MAX_DISTANCE || distance<1e-8) return false;
    // Conservatively reject any swept collision box touching a solid/changing cell.
    const radius=.36;
    for(let r=Math.floor(Math.min(y,ny)-radius);r<=Math.floor(Math.max(y,ny)+radius);r++) {
      for(let c=Math.floor(Math.min(x,nx)-radius);c<=Math.floor(Math.max(x,nx)+radius);c++) {
        if(r<0||r>=13||c<0||c>=15)return false;
        const i=r*15+c;
        for(const key of ['wall','brick','pushable','crate','fuse','fieldItem']) {
          if(Boolean(before[key]?.[i])!==Boolean(after[key]?.[i]))return false;
        }
        if(before.wall[i] || before.brick[i] || before.pushable?.[i] || before.fuse?.[i]>0)return false;
      }
    }
    return true;
  }
  function motion(before, after, fraction, enabled, events = []) {
    const curPos=Array.from(before.pos),smooth=[];
    for(let p=0;p<before.alive.length;p++) {
      smooth[p]=Boolean(enabled && canInterpolate(before,after,p,events));
      if(smooth[p]){curPos[p*2]=after.pos[p*2];curPos[p*2+1]=after.pos[p*2+1];}
    }
    return {humanPid:-1,tickMs:100,lastTickT:before.t*100,prevPos:Array.from(before.pos),curPos,
      fraction:enabled?Math.max(0,Math.min(1,fraction)):0,smooth,
      characters:['maomao','maomao'],playerPalettes:[null,'blue'],playerLabels:['模型 P0','对手 P1'],hideResult:true};
  }
  return { REWARDS, reward, componentNames, MAX_DISTANCE, canInterpolate, motion };
});
