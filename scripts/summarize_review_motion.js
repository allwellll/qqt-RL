'use strict';
// Publish bounded measurements, not the private per-frame samples or recordings.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('node:assert/strict');
const P=require('../web/review_protocol');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const [full,normal,output]=process.argv.slice(2);assert(full&&normal&&output,'FULL_CAPTURE NORMAL_CAPTURE OUTPUT');
const catalog=JSON.parse(fs.readFileSync('web/review-catalog.json')),row=catalog.replays.find(r=>r.diagnostic),doc=JSON.parse(fs.readFileSync('web/review-replays/'+row.file));
const report={schema:'qqt.replay-motion-audit/v1',replay_id:row.id,replay_sha256:row.sha256,simulator_sha256:doc.meta.simulator_sha256,units:'CSS screen pixels; deterministic probe coordinates are canvas pixels',definition:'Only safe adjacent-tick player positions interpolate; authoritative state/events/rewards remain integer-tick. A source-owned new bomb alone is not a player discontinuity.',runs:[],discrete_events:doc.events.filter(e=>['death','revive','control','control_end','pickup','drop','delivery'].includes(e.kind))};
for(const [root,speed]of [[full,4],[normal,1]])for(const key of ['desktop-smooth','desktop-discrete','mobile-smooth','mobile-discrete']){
  const folder=path.join(root,key),raw=fs.readFileSync(path.join(folder,'frames.json')),bytes=fs.readFileSync(path.join(folder,'summary.json')),frames=JSON.parse(raw),summary=JSON.parse(bytes),legalSnaps=[],unexpected=[...summary.anomalies];
  for(let i=1;i<frames.length;i++){
    const a=frames[i-1],b=frames[i];if(!a.playing||!b.playing)continue;
    for(const now of b.sprites){const prior=a.sprites.find(s=>s.pid===now.pid);if(!prior)continue;const pid=now.pid,distance=Math.hypot(now.screenX-prior.screenX,now.screenY-prior.screenY);
      if(a.smooth[pid]&&b.tick-a.tick<=1){const limit=.6*60*Math.max(0,b.now-a.now)/100*Math.max(a.scaleX,a.scaleY,b.scaleX,b.scaleY)+1e-5;if(distance>limit)unexpected.push({tick:b.tick,pid,kind:'excess_safe_frame_displacement',distance,limit});}
      if(key.endsWith('-smooth')&&distance>5&&!a.smooth[pid]){const reason=a.reasons?.[pid]||'unclassified';legalSnaps.push({tick:b.tick,pid,distance,reason});if(reason==='unclassified'||reason==='missing_adjacent_tick')unexpected.push({tick:b.tick,pid,kind:reason});}
    }
  }
  assert.deepEqual(unexpected,[],'motion anomalies');assert.deepEqual(summary.errors,[]);assert.deepEqual(summary.http,[]);
  report.runs.push({mode:key,speed,frames:summary.frames,first_tick:summary.first_tick,last_tick:summary.last_tick,actual_fps:summary.actual_fps,max_frame_screen_distance:summary.max_frame_screen_distance,max_continuous_frame_screen_distance:summary.max_continuous_frame_screen_distance,max_wall_gap_ms:summary.max_wall_gap_ms,css_scales:summary.css_scales,safe_intermediate_groups:summary.safe_intermediate_groups,probes:summary.probes,legal_snaps:legalSnaps,unexpected_jumps:unexpected,scheduler_gaps:summary.scheduler_gaps,pause_seek_step:summary.pause_seek_step,source_frames_sha256:sha(raw),source_summary_sha256:sha(bytes),console_errors:summary.errors,http_errors:summary.http});
}
assert(report.runs.filter(r=>r.speed===4).every(r=>r.last_tick===1200));assert(report.runs.filter(r=>r.speed===1&&r.mode.endsWith('-smooth')).every(r=>r.safe_intermediate_groups.every(n=>n>0)));
fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({runs:report.runs.length,unexpected_jumps:0,report_sha256:sha(fs.readFileSync(output)),bytes:fs.statSync(output).size}));
