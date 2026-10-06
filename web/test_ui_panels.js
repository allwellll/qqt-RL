'use strict';
const assert = require('assert');
const {mount} = require('./ui_panels');
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
mount({getElementById:element});
assert(element('advanced-settings').hidden);
click('advanced-open'); assert(element('password-dialog').open); assert.equal(focused,'settings-password');
element('settings-password').value='wrong'; submit(); assert(element('advanced-settings').hidden); assert.match(element('password-status').textContent,/口令不对/);
element('settings-password').value='demaxiya'; submit(); assert(!element('advanced-settings').hidden); assert.equal(element('settings-password').value,''); assert.equal(focused,'advanced-close');
element('advanced-settings').listeners.keydown({key:'Escape',preventDefault(){}}); assert(element('advanced-settings').hidden);
click('advanced-open'); assert(element('password-dialog').open); assert.equal(element('settings-password').value,'');
click('password-dialog-close'); assert(!element('password-dialog').open); assert.equal(focused,'advanced-open');
click('announcement-open'); assert(element('announcement-dialog').open);
element('announcement-dialog').listeners.click({target:element('announcement-dialog'),clientX:30,clientY:30}); assert(element('announcement-dialog').open,'padding inside dialog must not dismiss');
element('announcement-dialog').listeners.click({target:element('announcement-dialog'),clientX:1,clientY:1}); assert(!element('announcement-dialog').open); assert.equal(focused,'announcement-open');
click('announcement-open'); click('announcement-dialog-close'); assert(!element('announcement-dialog').open);
mount({getElementById:element}); assert(element('advanced-settings').hidden,'refresh always locks');
console.log('UI panels: wrong/correct password, re-lock, Escape, announcement backdrop/close and focus passed');
