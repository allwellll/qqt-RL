'use strict';
// A process uses one physics revision. Old replay hashes require the exact archive.
const fs=require('node:fs'),crypto=require('node:crypto'),path=require('node:path');
const revision=process.env.QQT_EVAL_SIMULATOR||'current';
const files={current:'web/sim-bun-native-v3-items.js',items:'web/sim-bun-native-v3-items.js',face:'web/sim-bun-native-v2-face.js','native-v2':'web/sim-bun-native-v2.js',legacy:'web/review-sim/bun-native-v2-20261104.js'};
if(!Object.hasOwn(files,revision))throw Error('QQT_EVAL_SIMULATOR must be current, items, face, native-v2 or legacy');
const file=files[revision],bytes=fs.readFileSync(path.resolve(__dirname,'..',file));
const sha=crypto.createHash('sha256').update(bytes).digest('hex');
if(revision==='legacy'&&sha!=='f6e346f3022640556efc2a988eb3b9b8e90abd5776af6ca1b4b1b0ee2acd6abc')throw Error('Archived simulator hash mismatch');
module.exports={...require(path.resolve(__dirname,'..',file)),SIMULATOR_FILE:file,SIMULATOR_SHA256:sha};
