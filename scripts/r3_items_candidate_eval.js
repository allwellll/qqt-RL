'use strict';
const fs=require('node:fs');
const original=require('./je05_web_hunter'),custom=require('./r3_items_inference'),load=original.loadModel;
original.loadModel=async file=>JSON.parse(fs.readFileSync(file)).meta.r3_input_encoding?custom.loadModel(file):load(file);
require('./r3_items_candidate_metrics').install(require('../web/sim-bun-native-v2'));
require('./bun_native_v2_eval');
