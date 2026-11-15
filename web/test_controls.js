'use strict';

const assert = require('assert');
const controls = require('./controls.js');

{
  const buttons = ['ArrowUp', 'Space'].map(key => ({ dataset: {controlKey:key}, handlers:{},
    addEventListener(name,fn){this.handlers[name]=fn;}, setPointerCapture(){} }));
  const held = new Set(), pressed = [];
  const release = controls.bindTouchControls({querySelectorAll:()=>buttons},held,key=>pressed.push(key));
  const event = pointerId => ({pointerId,preventDefault(){}});
  buttons[0].handlers.pointerdown(event(1)); buttons[0].handlers.pointerdown(event(2));
  buttons[1].handlers.pointerdown(event(3));
  assert.equal(controls.moveForHeld(held),0); assert(held.has('Space'));
  buttons[0].handlers.pointerup(event(1)); assert(held.has('ArrowUp'),'second pointer still holds direction');
  buttons[0].handlers.pointercancel(event(2)); assert(!held.has('ArrowUp'));
  buttons[1].handlers.lostpointercapture(event(3)); assert.equal(held.size,0);
  buttons[0].handlers.pointerdown(event(4)); release(); assert.equal(held.size,0);
  assert.deepEqual(pressed,['ArrowUp','Space','ArrowUp'],'two pointers holding the same key produce only one initial press');
}

{
  const slots=Array.from({length:7},(_,i)=>({item:i%2+1,count:i+2}));
  const before=JSON.stringify(slots);
  const pages=[0,1,2].map(p=>controls.touchItemPage(slots,p));
  assert.deepEqual(pages.map(p=>p.buttons.map(b=>b.key)),[['Digit1','Digit2','Digit3'],['Digit4','Digit5','Digit6'],['Digit7','','']]);
  assert.deepEqual(pages.flatMap(p=>p.buttons.filter(b=>b.slot).map(b=>b.slot)),slots,'all seven slots remain accessible');
  assert.equal(JSON.stringify(slots),before,'paging never mutates inventory');
  assert.equal(controls.touchItemPage([],2).page,0,'reset/empty inventory returns first page');
  assert.equal(controls.touchItemPage(slots.slice(0,3),2).page,0,'inventory shrink clamps to an accessible page');
}
{
  const stick = controls.createFloatingStick();
  assert.equal(stick.start(70, 90, 1000).direction, null, 'touch starts idle');
  assert.equal(stick.move(73, 90, 1008).direction, null, '3px stays below activation distance');
  assert.equal(stick.move(74, 90, 1016).direction, 'ArrowRight', '4px activates');
  stick.move(170, 90, 1024);
  assert.equal(stick.move(170, 190, 1032).direction, 'ArrowDown', 'downward stroke turns without recentering');
  assert.equal(stick.move(172, 190, 1040).direction, 'ArrowDown', '2px sideways jitter is ignored');
  const right = stick.move(173, 190, 1048);
  assert.equal(right.direction, 'ArrowRight', '3px horizontal stroke turns while far below initial center');
  assert(right.center.x < 173 && right.center.y === 190, 'dial follows behind current rightward intent');
  assert.equal(stick.move(173, 190, 1500).direction, 'ArrowRight', 'stationary finger keeps moving');
  stick.move(800, 190, 1508);
  assert.equal(stick.move(797, 190, 1516).direction, 'ArrowLeft', '3px reversal ignores accumulated travel');
  assert.equal(stick.end().direction, null, 'release stops immediately');
  assert.equal(stick.move(700, 190, 1524).direction, null, 'events after release cannot move');
  assert.equal(stick.start(400, 93, 1600).direction, null, 're-touch never restores stale direction');
  stick.move(404, 93, 1608);
  stick.move(404, 95, 1616);
  assert.equal(stick.move(404, 96, 1800).direction, 'ArrowDown', 'slow coherent strokes survive a pause between events');
  assert.equal(stick.move(404, 98, 1808).direction, 'ArrowDown', 'continued stroke keeps direction');
  stick.end(1900, true);
  assert.equal(stick.start(400, 93, 1901).direction, null, 'cancel clears input');
  assert.deepEqual(stick.start(700, 400, 2000).habit, {x:700,y:400}, 'new touch sets local feedback');
  stick.reset(); assert.equal(stick.start(1, 1, 2001).direction, null, 'reset clears input');
  for (const [dx, dy, key] of [[1,0,'ArrowRight'],[-1,0,'ArrowLeft'],[0,1,'ArrowDown'],[0,-1,'ArrowUp']]) {
    const slow = controls.createFloatingStick(); slow.start(0, 0, 0);
    for (let i = 1; i <= 16; i++) slow.move(dx*i/4, dy*i/4, i*8);
    assert.equal(slow.move(dx*4, dy*4, 136).direction, key, 'subpixel events accumulate into a deliberate stroke');
  }
  const diagonal = controls.createFloatingStick(); diagonal.start(0,0,0); diagonal.move(4,0,8);
  assert.equal(diagonal.move(14,10,16).direction,'ArrowRight','diagonal boundary retains current direction');
  assert.equal(diagonal.move(15,14,24).direction,'ArrowDown','clear downward trend turns');
  const jitter = controls.createFloatingStick(); jitter.start(0,0); jitter.move(100,0);
  for (let i = 0; i < 20; i++) {
    assert.equal(jitter.move(100,2).direction, 'ArrowRight', 'small down jitter does not turn');
    assert.equal(jitter.move(100,0).direction, 'ArrowRight', 'returning jitter does not accumulate into a turn');
  }
  const names = ['ArrowRight','ArrowDown','ArrowLeft','ArrowUp'];
  const rotations = [(x,y)=>[x,y], (x,y)=>[-y,x], (x,y)=>[-x,-y], (x,y)=>[y,-x]];
  for (let rotation = 0; rotation < 4; rotation++) for (const samples of [1,4,16,48]) {
    const sampled = controls.createFloatingStick(); sampled.start(500,500);
    let px = 0, py = 0;
    for (const [dx,dy,heading] of [[100,0,0],[0,60,1],[6,0,0],[-6,0,2],[0,-6,3]]) {
      for (let i = 1; i <= samples; i++) {
        const [x,y] = rotations[rotation](px+dx*i/samples, py+dy*i/samples);
        sampled.move(500+x,500+y);
      }
      px += dx; py += dy;
      const [x,y] = rotations[rotation](px,py);
      assert.equal(sampled.move(500+x,500+y).direction,names[(heading+rotation)%4],
        'corners and reversals work in every orientation across pointer sample rates');
    }
  }
  const held = new Set(['ArrowRight', 'touch:ArrowRight']);
  held.delete('touch:ArrowRight'); assert.equal(controls.moveForHeld(held), 3, 'touch release preserves keyboard');
  assert(!controls.MOVEMENT_KEYS.includes('touch:ArrowRight'), 'touch keys are not keyboard input');
}
assert.strictEqual(controls.moveForHeld(new Set(['ArrowUp'])), 0);
assert.strictEqual(controls.moveForHeld(new Set(['ArrowDown'])), 1);
assert.strictEqual(controls.moveForHeld(new Set(['ArrowLeft'])), 2);
assert.strictEqual(controls.moveForHeld(new Set(['ArrowRight'])), 3);
assert.strictEqual(controls.moveForHeld(new Set()), 4);
for (const first of controls.MOVEMENT_KEYS) for (const last of controls.MOVEMENT_KEYS) {
  if (first === last) continue;
  const held = new Set([first, last]);
  assert.equal(controls.moveForHeld(held), controls.MOVEMENT_DIRECTIONS[last], 'new direction wins');
  held.add(first);
  assert.equal(controls.moveForHeld(held), controls.MOVEMENT_DIRECTIONS[last], 'repeat cannot steal priority');
  held.delete(last);
  assert.equal(controls.moveForHeld(held), controls.MOVEMENT_DIRECTIONS[first], 'release restores held direction');
  held.clear();
  assert.equal(controls.moveForHeld(held), 4);
}
for (const key of ['KeyW', 'KeyA', 'KeyS', 'KeyD']) {
  assert.strictEqual(controls.moveForHeld(new Set([key])), 4, `${key} 不再控制移动`);
}
assert.deepStrictEqual(controls.BOMB_KEYS, ['Space', 'KeyW'], 'Space 与 W 都放泡');
for (const [code, direction] of Object.entries({Numpad5:0,Numpad1:2,Numpad2:1,Numpad3:3})) {
  assert.equal(controls.moveForHeld(new Set([code])),direction);
  assert(controls.MOVEMENT_KEYS.includes(code),'numpad movement must be intercepted');
  assert(!controls.ITEM_SLOT_KEYS.includes(code),'numpad must never request an item');
}
assert.deepEqual(controls.ITEM_SLOT_KEYS,['Digit1','Digit2','Digit3','Digit4','Digit5','Digit6','Digit7'],'main-row items stay unchanged');
assert.equal(controls.moveForHeld(new Set(['Digit1','Digit2','Digit3','Digit5','Numpad4','Numpad6','Numpad8'])),4);
assert.equal(controls.moveForHeld(new Set(['toString'])),4,'unknown keys cannot inherit directions');
console.log('网页方向键移动控制测试通过');

(async () => {
  const held = new Set();
  const target = { closest: () => null };
  const event = { code: 'KeyR', target, repeat: false };
  assert(controls.isRestartKey(event, held));
  assert(!controls.isRestartKey({ ...event, repeat: true }, held));
  held.add('KeyR'); assert(!controls.isRestartKey(event, held)); held.clear();
  for (const modifier of ['ctrlKey', 'metaKey', 'altKey']) assert(!controls.isRestartKey({ ...event, [modifier]: true }, held));
  for (const tag of ['input', 'textarea', 'contenteditable']) {
    assert(!controls.isRestartKey({ ...event, target: { closest: () => ({ tag }) } }, held), 'typing targets never restart');
  }
  let restarts = 0, done = true, release;
  const restart = controls.createRestartGate(async () => {
    restarts++;
    await new Promise(resolve => { release = resolve; });
    done = false;
  }, () => done);
  const a = restart(), b = restart(true), c = restart();
  assert.strictEqual(a, b); assert.strictEqual(b, c);
  await Promise.resolve(); assert.equal(restarts, 1, 'R plus button clicks coalesce while async reset is pending');
  release(); assert.equal(await a, true);
  assert.equal(await restart(true), false, 'stale settlement button cannot restart replacement match');
  assert.equal(restarts, 1);
  let attempts = 0;
  const failure = controls.createRestartGate(async () => { if (++attempts === 1) throw Error('load failed'); }, () => true);
  await assert.rejects(failure(), /load failed/);
  assert.equal(await failure(), true, 'failed restart does not permanently lock the gate');
  console.log('R typing/repeat/modifier protection, shared async restart and error recovery passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
