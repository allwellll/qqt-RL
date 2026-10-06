begin;

-- Keep the deployed validator, identity check and idempotency logic private.
do $$
begin
  if to_regprocedure('qqt_private.qqt_submit_result_core(jsonb)') is null then
    alter function public.qqt_submit_result(jsonb) set schema qqt_private;
    alter function qqt_private.qqt_submit_result(jsonb) rename to qqt_submit_result_core;
  end if;
end;
$$;
revoke all on function qqt_private.qqt_submit_result_core(jsonb) from public, anon, authenticated, service_role;
create index if not exists match_results_cohort_duration
on qqt_private.match_results(result, mode, map_id, difficulty, opponent, player_id, game_duration_ms);

create or replace function public.qqt_submit_result(p_payload jsonb) returns jsonb
language plpgsql security definer set search_path = '' as $$
declare
  progress jsonb; current_match qqt_private.match_results;
  sample_total bigint; better bigint; worse bigint;
begin
  progress := qqt_private.qqt_submit_result_core(p_payload);
  select * into strict current_match from qqt_private.match_results
    where client_match_id = (p_payload->>'client_match_id')::uuid;
  -- One sample per peer, current match for self: slower repeat wins rank honestly.
  with peers as (
    select m.player_id,
      case when current_match.result = 'win' then min(m.game_duration_ms)
        else max(m.game_duration_ms) end as duration
    from qqt_private.match_results m
    where m.result = current_match.result and m.mode = current_match.mode
      and m.map_id = current_match.map_id and m.difficulty = current_match.difficulty
      and m.opponent = current_match.opponent and m.player_id <> current_match.player_id
    group by m.player_id
  ), samples as (
    select duration from peers union all select current_match.game_duration_ms
  )
  select count(*),
    count(*) filter (where case when current_match.result = 'win'
      then duration < current_match.game_duration_ms else duration > current_match.game_duration_ms end),
    count(*) filter (where case when current_match.result = 'win'
      then duration > current_match.game_duration_ms else duration < current_match.game_duration_ms end)
    into sample_total, better, worse from samples;
  return progress || jsonb_build_object(
    'client_match_id', current_match.client_match_id,
    'match_rank', jsonb_build_object(
      'rank', better + 1, 'total', sample_total,
      'percentile', case when sample_total > 1 then round(100.0 * worse / (sample_total - 1), 2) else 0 end,
      'result', current_match.result, 'duration_ms', current_match.game_duration_ms,
      'mode', current_match.mode, 'map_id', current_match.map_id,
      'difficulty', current_match.difficulty, 'opponent', current_match.opponent,
      'order', case when current_match.result = 'win' then 'asc' else 'desc' end,
      'comparison', 'player-best-v1'
    ));
end;
$$;
comment on function public.qqt_submit_result(jsonb) is
  'Bounded untrusted client results. Same result/mode/map/difficulty/opponent; one best per peer, current match for self; competition ties; percentile strictly worse peers / peer count.';
revoke all on function public.qqt_submit_result(jsonb) from public;
grant execute on function public.qqt_submit_result(jsonb) to anon, authenticated, service_role;
notify pgrst, 'reload schema';
commit;
