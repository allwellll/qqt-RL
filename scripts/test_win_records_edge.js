'use strict';
const assert=require('assert'),{randomUUID}=require('crypto');
(async()=>{
  const {createSettlementHandler}=await import('../supabase/functions/submit-result/handler.mjs');
  const pid=randomUUID(),mid=randomUUID(),calls=[];let missing=false,rejected=false;
  const handler=createSettlementHandler({env:k=>({SUPABASE_URL:'https://mock',SUPABASE_SERVICE_ROLE_KEY:'server-only'})[k],
    createClient:()=>({rpc:async(name,args)=>{calls.push({name,args});
      if(name==='qqt_submit_result')return{data:{level:1,points:3,wins:1,games:1,client_match_id:mid}};
      if(name==='qqt_record_match_ip')return missing?{error:{code:'PGRST202'}}:rejected?{error:{code:'22023'}}:
        {data:{recorded:true,ip_display:'12.*.*.78',raw_ip:'12.34.56.78',player_secret:'must-not-leak'}};
      return{data:{recorded:true,ip_display:'12.*.*.78'}};}})});
  const request=()=>new Request('https://edge',{method:'POST',headers:{'x-forwarded-for':'12.34.56.78'},
    body:JSON.stringify({p_payload:{player_id:pid,client_match_id:mid}})});
  let response=await(await handler(request())).json();
  assert.equal(calls[1].name,'qqt_record_match_ip','RED: source IP must be bound to the authenticated match');
  assert.deepEqual(calls[1].args,{p_player_id:pid,p_client_match_id:mid,p_raw_ip:'12.34.56.78'});
  assert.equal(response.ip_display,'12.*.*.78');assert(!JSON.stringify(response).includes('raw_ip'));assert(!JSON.stringify(response).includes('must-not-leak'));
  calls.length=0;missing=true;response=await(await handler(request())).json();
  assert.equal(calls[2].name,'qqt_record_player_ip','old database fallback keeps the existing private metadata contract');
  assert.equal(response.network_metadata_recorded,true);
  calls.length=0;missing=false;rejected=true;response=await(await handler(request())).json();
  assert.equal(calls.length,2);assert.equal(response.network_metadata_recorded,false,'business rejection never falls back to another writer');
  console.log('Win-record Edge: exact match source binding, masking, missing-RPC fallback and rejected metadata passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
