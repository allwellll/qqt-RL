'use strict';

const fs = require('fs');
const Replay = require('../web/replay.js');
const { makeDocument, sha } = require('./generate_je09_checkpoint_replay.js');

const [modelPath, replayPath] = process.argv.slice(2);
if (!modelPath || !replayPath) throw new Error('usage: node scripts/verify_je09_checkpoint_replay.js MODEL.json REPLAY.json');
const actualBytes = fs.readFileSync(replayPath);
const actual = Replay.validateReplay(JSON.parse(actualBytes));
const expected = makeDocument(modelPath);
const actualComparable = { ...actual, meta: { ...actual.meta, source_hashes: undefined } };
const expectedComparable = { ...expected, meta: { ...expected.meta, source_hashes: undefined } };
if (JSON.stringify(actualComparable) !== JSON.stringify(expectedComparable)) {
  throw new Error('replay actions or summary differ from deterministic model re-simulation');
}
for (const [source, digest] of Object.entries(actual.meta.source_hashes || {})) {
  if (!/^[0-9a-f]{64}$/.test(digest)) throw new Error(`invalid historical source hash: ${source}`);
}
console.log(JSON.stringify({ passed: true, ticks: actual.summary.ticks,
  replay_sha256: sha(actualBytes), model_json_sha256: actual.meta.model_json_sha256,
  checkpoint_sha256: actual.meta.checkpoint_sha256 }));
