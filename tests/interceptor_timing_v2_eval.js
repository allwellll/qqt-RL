'use strict';
const original=require('../web/bun_interceptor_bot'),candidate=require('./interceptor_timing_v2');
original.BunInterceptorBot=candidate.BunInterceptorBot;original.VERSION=candidate.VERSION;
require('../scripts/r4_interceptor_eval');
