'use strict';
const assert = require('assert');
const {mount} = require('./ui_panels');
const device = require('./device_auth');
const crypto = require('crypto');
const saved = new Map();
const storage = { getItem: k => saved.get(k), setItem: (k,v) => saved.set(k,v), removeItem: k => saved.delete(k) };
const fixture = 'test-only-password';
const verifier = crypto.createHash('sha256').update(`${device.VERSION}:${fixture}`).digest('hex');
const makeAuth = version => device.create(storage, crypto.webcrypto, version, verifier);
let auth = makeAuth();
const elements = new Map(); let focused;
function element(id) {
  if (!elements.has(id)) elements.set(id,{hidden:false,value:'',textContent:'',open:false,listeners:{},attrs:{},
    addEventListener(k,fn){this.listeners[k]=fn;}, setAttribute(k,v){this.attrs[k]=v;}, focus(){focused=id;},
    showModal(){this.open=true;}, close(){this.open=false;this.listeners.close?.();},
    querySelector(){return element(id+'-close');}, getBoundingClientRect(){return {left:10,right:200,top:10,bottom:200};}});
  return elements.get(id);
}
const click=id=>element(id).listeners.click({});
const submit=()=>element('password-form').listeners.submit({preventDefault(){}});
(async () => {
mount({getElementById:element}, {auth});
assert(element('advanced-settings').hidden);
click('advanced-open'); assert(element('password-dialog').open); assert.equal(focused,'settings-password');
element('settings-password').value='wrong'; await submit(); assert(element('advanced-settings').hidden); assert.match(element('password-status').textContent,/口令不对/); assert.equal(saved.size,0);
element('settings-password').value=fixture; await submit(); assert(!element('advanced-settings').hidden); assert.equal(element('settings-password').value,''); assert.equal(focused,'advanced-close');
element('advanced-settings').listeners.keydown({key:'Escape',preventDefault(){}}); assert(element('advanced-settings').hidden);
click('advanced-open'); assert(!element('password-dialog').open); assert(!element('advanced-settings').hidden);
click('advanced-forget'); click('advanced-open'); assert(element('password-dialog').open);
click('password-dialog-close'); assert(!element('password-dialog').open); assert.equal(focused,'advanced-open');
click('announcement-open'); assert(element('announcement-dialog').open);
element('announcement-dialog').listeners.click({target:element('announcement-dialog'),clientX:30,clientY:30}); assert(element('announcement-dialog').open,'padding inside dialog must not dismiss');
element('announcement-dialog').listeners.click({target:element('announcement-dialog'),clientX:1,clientY:1}); assert(!element('announcement-dialog').open); assert.equal(focused,'announcement-open');
click('announcement-open'); click('announcement-dialog-close'); assert(!element('announcement-dialog').open);
await auth.verify(fixture);
auth = makeAuth(); mount({getElementById:element}, {auth}); click('advanced-open'); assert(!element('advanced-settings').hidden,'refresh remembers');
assert(makeAuth().authorized(),'new tab remembers');
assert(!makeAuth('qqt.advanced/v2').authorized(),'version change invalidates');
assert.equal(saved.size,0);
await auth.verify(fixture); auth.forget(); assert(!makeAuth().authorized());
await auth.verify(fixture); assert(!device.create(storage,crypto.webcrypto,device.VERSION,'0'.repeat(64)).authorized(),'validation rule change invalidates');
const blocked=device.create({getItem(){throw Error('storage blocked');},setItem(){throw Error('storage blocked');},removeItem(){throw Error('storage blocked');}},crypto.webcrypto,device.VERSION,verifier);
assert(!blocked.authorized()); assert(await blocked.verify(fixture)); assert(blocked.authorized()); blocked.forget(); assert(!blocked.authorized());
assert(!JSON.stringify([...saved]).includes(fixture));
console.log('UI panels: first prompt, wrong/correct, refresh, new tab, forget, version invalidation, dialogs passed');
})().catch(e => { console.error(e); process.exitCode=1; });
