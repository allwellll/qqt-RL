import subprocess


def test_policy_opponent_uses_await_and_independent_opponent_rng():
    code=r"""
    const {transform}=require('./scripts/r3_sp_fixed_history_eval_20261031');
    const src="const loaded=await inference.loadModel(modelFile);const actorRng=Q.mulberry32(1),opponentRng=Q.mulberry32(2),hunter=authority.createHunter('hard');const a=await loaded.model.act(s,team,actorRng),bot=hunter.adapter.act(obs,enemy,opponentRng);";
    const out=transform(src);require('node:assert/strict').match(out,/bot=await hunter.adapter.act\(obs,enemy,opponentRng\)/);
    require('node:assert/strict').match(out,/fixedOpponent.model.act\(obs.metadata.sim,pid,rng\)/);
    new (Object.getPrototypeOf(async function(){}).constructor)('inference','Q','s','team','obs','enemy',out);
    require('node:assert/strict').throws(()=>transform('changed source'));
    """
    subprocess.run(['node','-e',code],check=True,capture_output=True,text=True)


def test_fixed_score_elo_and_kill_denominator():
    from scripts.r3_sp_fixed_history_owner_20261031 import summary
    rows=[dict(seat=0,winner=w,causal_kills=[k,0],deaths=[1,0]) for w,k in [(0,2),(0,1),(-1,0),(1,1)]]
    r=summary(rows,0)
    assert r['games']==4 and r['score']==.625 and r['causal_kills_per_game']==1.
    assert 88<r['elo_difference']<89
    assert summary(rows[:1],0)['elo_difference'] is None


def test_real_sp55_namespace_is_discovered_for_bot_and_fixed_history(tmp_path,monkeypatch):
    from scripts import r3_selfplay_eval_20261031 as bot
    from scripts import r3_sp_fixed_history_owner_20261031 as fixed
    from scripts.population_run_state import atomic_json
    monkeypatch.setattr(bot,'OUT',tmp_path);monkeypatch.setattr(fixed,'OUT',tmp_path)
    source={'checkpoint':'parent','policy_sha256':'parent','update':3040}
    atomic_json(tmp_path/'REGISTRATION.json',{'parent_source':source})
    folder=tmp_path/'formal/r3_sp55_c000000'
    atomic_json(folder/'eval_queue/u003120.json',dict(source,checkpoint='new',update=3120))
    atomic_json(folder/'FIRST_FORMAL_UPDATE.json',{'source':dict(source,checkpoint='first',update=3043)})
    assert any(r['source']['update']==3120 for r in bot.requests())
    assert {r['source']['update'] for r in fixed.requests()}=={3040,3043,3120}
