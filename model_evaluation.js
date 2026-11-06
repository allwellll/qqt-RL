'use strict';
const fs = require('node:fs'), path = require('node:path'), Module = require('node:module');
const config = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
Object.assign(process.env, config.env || {});
process.argv = [process.execPath, 'model_evaluation.js', ...(config.args || [])];
Module._load(path.resolve(config.script), null, true);
