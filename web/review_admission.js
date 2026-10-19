'use strict';
// Build-time gate: historical schema compatibility never grants publication.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),zlib=require('zlib'),P=require('./review_protocol'),M=require('./review_metrics');
const SCHEMA='qqt.review-admission/v1',sha=b=>crypto.createHash('sha256').update(b).digest('hex');
function validate(doc){P.validate(doc);M.validate(doc);if(doc.meta.diagnostic_admission?.schema!==SCHEMA||!doc.metrics||doc.meta.diagnostic_admission.status!=='qualified')throw Error('diagnostic admission: missing complete sealed reward/Critic evidence');if(doc.meta.comparison_model_id){const models=JSON.parse(fs.readFileSync(path.join(__dirname,'review-evidence/good-models-20261019/models.json'))).models,m=models.find(m=>m.id===doc.meta.comparison_model_id);if(!m||m.checkpoint_sha256!==doc.meta.checkpoint_sha256||m.model_sha256!==doc.meta.model_sha256||['critic_sha256','state_sha256','network_sha256','gamma'].some(k=>m.critic[k]!==doc.meta.metrics[k])||JSON.stringify(m.reward_hashes)!==JSON.stringify(doc.meta.metrics.reward_hashes))throw Error('diagnostic checkpoint / Critic / Reward pairing mismatch');}return true;}
function catalogGate(catalog,root=__dirname){
 const rows=[...catalog.replays,...catalog.history_replays||[]];if(!rows.length)throw Error('diagnostic catalog must not be empty');
 const critic=JSON.parse(fs.readFileSync(path.join(root,'review-evidence/metrics-20261018/critic.json')));
 for(const row of rows){const bytes=fs.readFileSync(path.join(root,'review-replays',row.file)),doc=JSON.parse(bytes);validate(doc);if(!doc.meta.comparison_model_id){if(doc.meta.metrics.gamma!==critic.gamma)throw Error('diagnostic admission gamma mismatch');for(const key of ['checkpoint_sha256','critic_sha256','state_sha256','network_sha256'])if(doc.meta.metrics[key]!==critic[key])throw Error('diagnostic admission mismatched checkpoint/Critic');}if(['checkpoint_sha256','model_sha256','seed','scene','comparison_model_id'].some(k=>row[k]!==doc.meta[k])||row.id!==doc.meta.id||row.sha256!==sha(bytes)||row.bytes!==bytes.length||row.gzip_bytes!==zlib.gzipSync(bytes,{level:9}).length)throw Error('diagnostic admission hash / size');}
 return rows.length;
}
if(require.main===module)console.log('Qualified complete diagnostic recordings:',catalogGate(JSON.parse(fs.readFileSync(path.join(__dirname,'review-catalog.json')))));
module.exports={SCHEMA,validate,catalogGate};
