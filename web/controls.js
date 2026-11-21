'use strict';

(function controlsFactory(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.QQTControls = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function buildControls() {
  // Physical codes stay stable when NumLock changes key values (5/Clear,
  // 1/End, 2/ArrowDown, 3/PageDown). Keep aliases separate in held state so
  // releasing one key never releases another key with the same direction.
  const MOVEMENT_DIRECTIONS = Object.freeze({
    ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3,
    Numpad5: 0, Numpad1: 2, Numpad2: 1, Numpad3: 3,
  });
  const MOVEMENT_KEYS = Object.keys(MOVEMENT_DIRECTIONS);
  // 原版用 Ctrl 放道具，但浏览器里 Ctrl+W/R 会关闭/刷新页面，改用 E / Shift。
  const ITEM_KEYS = ['KeyE', 'ShiftLeft', 'ShiftRight'];
  const BOMB_KEYS = ['Space', 'KeyW'];
  // 数字键 1-7 对应道具栏格位；E / Shift 使用第 1 格。
  const ITEM_SLOT_KEYS = ['Digit1', 'Digit2', 'Digit3', 'Digit4', 'Digit5', 'Digit6', 'Digit7'];

  function moveForHeld(held) {
    let move = 4;
    for (const key of held) {
      const code = key.startsWith('touch:') ? key.slice(6) : key;
      if (Object.hasOwn(MOVEMENT_DIRECTIONS, code)) move = MOVEMENT_DIRECTIONS[code];
    }
    return move;
  }

  function isTypingTarget(target) {
    return !!(target && target.closest && target.closest('input, textarea, [contenteditable]:not([contenteditable="false"])'));
  }

  function isRestartKey(event, held) {
    return event.code === 'KeyR' && !event.repeat && !held.has('KeyR') &&
      !event.ctrlKey && !event.metaKey && !event.altKey && !isTypingTarget(event.target);
  }

  // Coalesce keyboard/button requests while the existing reset (including map load)
  // is in progress. Completed-match buttons cannot restart the replacement match.
  function createRestartGate(reset, isFinished) {
    let pending = null;
    return function restart(finishedOnly = false) {
      if (pending) return pending;
      if (finishedOnly && !isFinished()) return Promise.resolve(false);
      pending = Promise.resolve().then(reset).then(() => true).finally(() => { pending = null; });
      return pending;
    };
  }

  // Pagination changes only visible shortcuts, never the authoritative inventory.
  function touchItemPage(slots, requestedPage = 0) {
    const pages = Math.max(1, Math.ceil(Math.min(slots.length, ITEM_SLOT_KEYS.length) / 3));
    const page = Math.max(0, Math.min(pages - 1, Number.isInteger(requestedPage) ? requestedPage : 0));
    return {page, pages, buttons: Array.from({length: 3}, (_, i) => {
      const index = page * 3 + i;
      return {index, number: index < ITEM_SLOT_KEYS.length ? index + 1 : null,
        key: ITEM_SLOT_KEYS[index] || '', slot: slots[index] || null};
    })};
  }

  const STICK_KEYS = ['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp'];
  // Radial intent is measured from the real floating centre, never event deltas.
  function createFloatingStick({radius = 48, deadzone = radius * .18, hysteresis = 58, diagonalMin = 25, diagonalMax = 65, diagonalMode = 'secondary'} = {}) {
    let center = null, habit = null, direction = null, secondary = null, changedAt = 0;
    const snapshot = () => ({center: center && {...center}, habit: habit && {...habit}, direction, secondary, changedAt});
    return {
      start(x, y, now = performance.now(), initialDirection = null) {
        center = habit = {x,y}; direction = initialDirection; secondary = null; changedAt = now;
        return snapshot();
      },
      move(x, y, now = performance.now()) {
        if (!center) return snapshot();
        let dx=x-center.x, dy=y-center.y, length=Math.hypot(dx,dy);
        if(length>radius){center={x:x-dx*radius/length,y:y-dy*radius/length};dx=x-center.x;dy=y-center.y;length=radius;}
        const old=direction, oldSecondary=secondary;
        direction=secondary=null;
        if(length>=deadzone){
          const angle=Math.atan2(dy,dx), index=STICK_KEYS.indexOf(old);
          const delta=index<0?Infinity:Math.abs(Math.atan2(Math.sin(angle-index*Math.PI/2),Math.cos(angle-index*Math.PI/2)));
          const nearest=Math.abs(dx)>=Math.abs(dy) ? (dx<0?'ArrowLeft':'ArrowRight') : (dy<0?'ArrowUp':'ArrowDown');
          direction=delta<=hysteresis*Math.PI/180?old:nearest;
          const diagonal=Math.atan2(Math.abs(dy),Math.abs(dx))*180/Math.PI;
          if(diagonalMode==='preserve' && diagonal>=diagonalMin && diagonal<=diagonalMax){
            const previous=old;
            if(previous==='ArrowLeft'||previous==='ArrowRight'||previous==='ArrowUp'||previous==='ArrowDown')direction=previous;
            else direction=nearest;
          } else if(diagonal>=diagonalMin&&diagonal<=diagonalMax){
            const horizontal=dx<0?'ArrowLeft':'ArrowRight',vertical=dy<0?'ArrowUp':'ArrowDown';
            secondary=direction===horizontal?vertical:horizontal;
          }
        }
        if(old!==direction||oldSecondary!==secondary)changedAt=now;
        return snapshot();
      },
      end(now = performance.now()) { habit=center&&{...center};center=null;direction=secondary=null;changedAt=now;return snapshot(); },
      reset(now = performance.now()) { center=habit=direction=secondary=null;changedAt=now;return snapshot(); },
    };
  }

  // Read-only native projection-step probe. _nativeResolve may edit its state argument,
  // so give it a private copy. No step, touch clock, RNG, push or field contact.
  function nativeDirectionReachable(sim, pid, move) {
    if(!sim.alive[pid] || sim.trapped?.[pid]>0 || typeof sim._nativeResolve!=='function')return false;
    const original=sim.nativeMove?.[pid];
    const state={...original,wallExit:original?.wallExit?{...original.wallExit}:null};
    const x=Math.round(sim.pos[pid*2+1]*40),y=Math.round(sim.pos[pid*2]*40);
    const distance=Math.max(1,Math.min(40,Math.ceil(sim._nativeSpeedPx(pid)*.025)));
    const [nx,ny]=sim._nativeResolve(state,x,y,move,distance,true);
    // Existing corner alignment counts as reachable; actual frameStep owns the move.
    return nx!==x||ny!==y;
  }

  function createTouchSteering({bufferMs=160}={}) {
    let signature=null, travel=null, pending=null, until=0;
    const clear=()=>{signature=travel=pending=null;until=0;};
    return {clear, select(state,now,reachable) {
      if(!state?.direction){clear();return null;}
      const candidates=[state.direction,state.secondary].filter(Boolean);
      const perpendicular=(a,b)=>a&&b&&((MOVEMENT_DIRECTIONS[a]<2)!==(MOVEMENT_DIRECTIONS[b]<2));
      const next=candidates.join(':');
      if(signature!==next){
        signature=next;
        const target=candidates.find(key=>perpendicular(travel,key));
        pending=target||null;until=(state.changedAt??now)+bufferMs;
        if(!target)travel=state.direction; // Includes immediate opposite-direction input.
      }
      if(pending&&now<=until){
        if(reachable(MOVEMENT_DIRECTIONS[pending])){travel=pending;pending=null;return travel;}
        if(travel&&reachable(MOVEMENT_DIRECTIONS[travel]))return travel;
      } else pending=null;
      // A completed diagonal turn stays chosen until intent changes, not alternate every frame.
      if(travel&&candidates.includes(travel)&&reachable(MOVEMENT_DIRECTIONS[travel]))return travel;
      if(reachable(MOVEMENT_DIRECTIONS[state.direction]))return travel=state.direction;
      if(state.secondary&&reachable(MOVEMENT_DIRECTIONS[state.secondary]))return travel=state.secondary;
      return state.direction; // Still face a wall/bubble; never fabricate movement.
    }};
  }

  function bindFloatingDirection(zone, held, options = {}) {
    let pointer = null, stick = null, lastDirection = null, current = null;
    let sampled = false, pendingTap = null, tapSeconds = null;
    const pointers=new Map();
    let visibleCenter=null, lastTouch=null;
    let steering=createTouchSteering({bufferMs: options.bufferMs ?? 160});
    const dial = zone.querySelector('.touch-stick');
    const paint = state => {
      current=state;
      if(!state.direction)steering.clear();
      for (const key of STICK_KEYS) held.delete('touch:' + key);
      if (state.direction) held.add('touch:' + state.direction);
      const rect = zone.getBoundingClientRect(), p = state.center || visibleCenter;
      if(state.center)visibleCenter={...state.center};
      if (p) { dial.style.left = (p.x - rect.left) + 'px'; dial.style.top = (p.y - rect.top) + 'px'; }
      else { dial.style.left = ''; dial.style.top = ''; }
      zone.dataset.direction = state.direction || '';
      zone.dataset.secondary = state.secondary || '';
      zone.dataset.resolved = '';
      zone.dataset.active = pointer == null ? 'false' : 'true';
      const vector=state.center;
      zone.dataset.rawAngle=vector?String(Math.round(Math.atan2(vector.y,vector.x)*180/Math.PI)):'';
      zone.dataset.rawRadius=vector?String(Math.round(Math.hypot(vector.x,vector.y))):'';
      if (state.direction && state.direction !== lastDirection) {
        try { globalThis.navigator?.vibrate?.(8); } catch (_) { /* Optional feedback. */ }
      }
      lastDirection = state.direction;
    };
    const stale=(event,record)=>Number.isFinite(event.timeStamp)&&event.timeStamp<record.since;
    const relativeHeading=(x,y,center,previous,direction)=>{
      const dx=x-center.x,dy=y-center.y,angle=Math.atan2(Math.abs(dy),Math.abs(dx))*180/Math.PI;
      const horizontal=dx<0?'ArrowLeft':'ArrowRight',vertical=dy<0?'ArrowUp':'ArrowDown';
      if(angle<35)return horizontal;
      if(angle>55)return vertical;
      // Only an ambiguous diagonal uses a clear inter-touch displacement or
      // the active direction. Strong cardinal positions always win afresh.
      if(previous){
        const px=x-previous.x,py=y-previous.y;
        if(Math.hypot(px,py)>=8){
          if(Math.abs(px)>=Math.abs(py)*1.5&&(px<0?'ArrowLeft':'ArrowRight')===horizontal)return horizontal;
          if(Math.abs(py)>=Math.abs(px)*1.5&&(py<0?'ArrowUp':'ArrowDown')===vertical)return vertical;
        }
      }
      if(direction===horizontal||direction===vertical)return direction;
      return Math.abs(dx)>=Math.abs(dy)?horizontal:vertical;
    };
    const take=(id,record,center,hint=null)=>{
      const old=pointers.get(pointer);
      if(old){old.anchorX=old.x;old.anchorY=old.y;}
      pointer=id;sampled=false;pendingTap=null;steering.clear();
      stick.start(center.x,center.y,performance.now(),hint);
      paint(stick.move(record.x,record.y));
    };
    zone.addEventListener('pointerdown', event => {
      if(event.button!==0||pointers.has(event.pointerId))return;
      const rect=zone.getBoundingClientRect(),x=event.clientX,y=event.clientY;
      if(x<rect.left+8||x>rect.right-8||y<rect.top+8||y>rect.bottom-8)return;
      event.preventDefault();
      const radius=dial.getBoundingClientRect().width/2||48;
      if(!stick)stick=createFloatingStick({radius, deadzone: radius * (options.deadzoneRatio ?? .18), hysteresis: options.hysteresisDeg ?? 58, diagonalMin: options.diagonalMin ?? 25, diagonalMax: options.diagonalMax ?? 65, diagonalMode: options.diagonalMode ?? 'secondary'});
      const center=visibleCenter||{x:rect.left+rect.width/2,y:rect.top+rect.height/2};
      const record={x,y,anchorX:x,anchorY:y,since:event.timeStamp??0};
      const old=pointers.get(pointer),previous=old||(lastTouch&&performance.now()-lastTouch.time<=200?lastTouch:null);
      pointers.set(event.pointerId,record);zone.setPointerCapture(event.pointerId);
      if(Math.hypot(x-center.x,y-center.y)<radius*(options.deadzoneRatio ?? .18)){
        // Neutral new finger can start a drag, but cannot interrupt an owner.
        if(pointer==null&&!current?.direction)take(event.pointerId,record,center);
        return;
      }
      take(event.pointerId,record,center,relativeHeading(x,y,center,previous,current?.direction));
    });
    zone.addEventListener('pointermove', event => {
      const record=pointers.get(event.pointerId);
      if(!record||stale(event,record))return;
      event.preventDefault();
      const samples=event.getCoalescedEvents?.();
      let state;
      for(const sample of samples?.length?samples:[event]){
        record.x=sample.clientX;record.y=sample.clientY;
        if(event.pointerId!==pointer){
          const radius=dial.getBoundingClientRect().width/2||48;
          if(Math.hypot(record.x-record.anchorX,record.y-record.anchorY)<Math.max(12,radius*.25))continue;
          // A suspended finger must deliberately drag; rebase at its suspension
          // point so its old absolute offset cannot pull the direction back.
          take(event.pointerId,record,{x:record.anchorX,y:record.anchorY});
        }
        state=stick.move(record.x,record.y);
      }
      if(state)paint(state);
    });
    const clear = () => {
      const ids=[...pointers.keys()];pointers.clear();pointer=null;
      pendingTap=null;tapSeconds=null;sampled=false;visibleCenter=lastTouch=null;
      if(stick)paint(stick.reset());
      stick=null;
      steering=createTouchSteering({bufferMs: options.bufferMs ?? 160});
      for(const id of ids)if(zone.hasPointerCapture?.(id))zone.releasePointerCapture(id);
    };
    const release=event=>{
      const record=pointers.get(event.pointerId);
      if(!record||stale(event,record))return;
      if(event.type==='pointercancel'){clear();return;}
      if(event.type==='lostpointercapture'){
        if(event.pointerId===pointer)clear();else pointers.delete(event.pointerId);
        return;
      }
      pointers.delete(event.pointerId);
      if(event.pointerId!==pointer)return;
      pendingTap=!sampled&&current?.direction?{state:{...current},expires:performance.now()+100}:null;
      lastTouch={x:record.x,y:record.y,time:performance.now()};
      pointer=null;paint(stick.reset());
      // Other fingers stay suspended. Releasing the newest finger never restores them.
    };
    for(const name of ['pointerup','pointercancel','lostpointercapture'])zone.addEventListener(name,release);
    globalThis.addEventListener?.('resize', clear);
    clear.moveFor=(sim,pid,now)=>{
      tapSeconds=null;
      let last=null;for(const key of held)if(Object.hasOwn(MOVEMENT_DIRECTIONS,key.startsWith('touch:')?key.slice(6):key))last=key;
      const forced=!sim.alive[pid]||sim.trapped?.[pid]>0||sim.playerMoveDirection(pid,4)!==4;
      if(!last && pendingTap){
        const tap=pendingTap;pendingTap=null;
        if(now<=tap.expires&&!forced){
          steering.clear();
          const key=steering.select(tap.state,now,mv=>nativeDirectionReachable(sim,pid,mv));
          steering.clear();tapSeconds=1/60;
          zone.dataset.resolved=key||'';
          return key?MOVEMENT_DIRECTIONS[key]:4;
        }
      }
      if(!last?.startsWith('touch:')){pendingTap=null;steering.clear();zone.dataset.resolved='';return moveForHeld(held);}
      sampled=true;
      // Forced movement stays owned by the simulator and cannot queue a delayed turn.
      if(forced){pendingTap=null;steering.clear();return moveForHeld(held);}
      const key=steering.select(current,now,mv=>nativeDirectionReachable(sim,pid,mv));
      zone.dataset.resolved=key||'';
      return key?MOVEMENT_DIRECTIONS[key]:4;
    };
    clear.frameSeconds=dt=>{const seconds=tapSeconds;tapSeconds=null;return seconds??dt;};
    return clear;
  }

  function bindTouchControls(root, held, onPress, options = {}) {
    const active = new Map();
    const zone = root.matches?.('[data-floating-direction]') ? root : root.querySelector?.('[data-floating-direction]');
    const clearDirection = zone ? bindFloatingDirection(zone, held, options) : () => {};
    const release = event => {
      const key = active.get(event.pointerId);
      active.delete(event.pointerId);
      if (key && ![...active.values()].includes(key)) held.delete(key);
    };
    // Restrict browser gesture suppression to the game controls.
    for (const name of ['contextmenu', 'selectstart', 'dragstart'])
      root.addEventListener?.(name, event => event.preventDefault());
    if (zone && zone !== root) {
      for (const name of ['contextmenu', 'selectstart', 'dragstart'])
        zone.addEventListener?.(name, event => event.preventDefault());
    }
    for (const button of root.querySelectorAll('[data-control-key]')) {
      button.addEventListener('pointerdown', event => {
        if (event.button != null && event.button !== 0) return;
        event.preventDefault();
        const key = button.dataset.controlKey;
        if (!key || button.getAttribute?.('aria-disabled') === 'true' || active.has(event.pointerId)) return;
        const alreadyHeld = [...active.values()].includes(key);
        active.set(event.pointerId, key); held.add(key);
        button.setPointerCapture(event.pointerId);
        if (!alreadyHeld) onPress(key);
      });
      for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) button.addEventListener(name, release);
    }
    const clear=() => { clearDirection(); for (const key of active.values()) held.delete(key); active.clear(); };
    clear.moveFor=clearDirection.moveFor||(()=>moveForHeld(held));
    clear.frameSeconds=clearDirection.frameSeconds||(dt=>dt);
    return clear;
  }

  return { MOVEMENT_KEYS, MOVEMENT_DIRECTIONS, ITEM_KEYS, ITEM_SLOT_KEYS, BOMB_KEYS, moveForHeld,
    isTypingTarget, isRestartKey, createRestartGate, touchItemPage, createFloatingStick, nativeDirectionReachable, createTouchSteering, bindTouchControls };
});
