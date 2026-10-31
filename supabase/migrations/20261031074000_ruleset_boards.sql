begin;
-- Preserve all historic results. Missing versions are explicitly legacy v1.
alter table qqt_private.match_results add column if not exists ruleset_version text not null default 'stock-window-v1';
do $$ begin
 if not exists(select 1 from pg_constraint where conname='match_results_ruleset_version_check' and conrelid='qqt_private.match_results'::regclass) then
  alter table qqt_private.match_results add constraint match_results_ruleset_version_check check(ruleset_version in ('stock-window-v1','bun-native-v2'));
 end if;
 if to_regprocedure('qqt_private.qqt_submit_result_before_ruleset(jsonb)') is null then
  alter function public.qqt_submit_result(jsonb) set schema qqt_private;
  alter function qqt_private.qqt_submit_result(jsonb) rename to qqt_submit_result_before_ruleset;
 end if;
end $$;
revoke all on function qqt_private.qqt_submit_result_before_ruleset(jsonb) from public,anon,authenticated,service_role;
create index if not exists match_results_ruleset_cohort on qqt_private.match_results(ruleset_version,result,mode,map_id,difficulty,opponent,player_id,game_duration_ms);

create or replace function public.qqt_submit_result(p_payload jsonb) returns jsonb
language plpgsql security definer set search_path='' as $$
declare response jsonb; version text; mid uuid; current_match qqt_private.match_results;
 sample_total bigint; better bigint; worse bigint;
begin
 if p_payload is null or jsonb_typeof(p_payload)<>'object' then raise exception 'invalid payload' using errcode='22023'; end if;
 version:=case when p_payload ? 'ruleset_version' then p_payload->>'ruleset_version' else 'stock-window-v1' end;
 if version is null or version not in ('stock-window-v1','bun-native-v2') then raise exception 'invalid ruleset version' using errcode='22023'; end if;
 mid:=(p_payload->>'client_match_id')::uuid;
 -- Same lock order as the preserved validator: identity then match.
 perform pg_advisory_xact_lock(hashtextextended(p_payload->>'player_id',0));
 perform pg_advisory_xact_lock(hashtextextended(mid::text,1));
 if exists(select 1 from qqt_private.match_results where client_match_id=mid and ruleset_version<>version) then
  raise exception 'ruleset mismatch on retry' using errcode='22023';
 end if;
 response:=qqt_private.qqt_submit_result_before_ruleset(p_payload-'ruleset_version');
 update qqt_private.match_results set ruleset_version=version where client_match_id=mid;
 select * into strict current_match from qqt_private.match_results where client_match_id=mid;
 with peers as (
  select m.player_id,case when current_match.result='win' then min(m.game_duration_ms) else max(m.game_duration_ms) end duration
  from qqt_private.match_results m where m.ruleset_version=version and m.result=current_match.result
   and m.mode=current_match.mode and m.map_id=current_match.map_id and m.difficulty=current_match.difficulty
   and m.opponent=current_match.opponent and m.player_id<>current_match.player_id group by m.player_id
 ), samples as (select duration from peers union all select current_match.game_duration_ms)
 select count(*),count(*) filter(where case when current_match.result='win' then duration<current_match.game_duration_ms else duration>current_match.game_duration_ms end),
 count(*) filter(where case when current_match.result='win' then duration>current_match.game_duration_ms else duration<current_match.game_duration_ms end)
 into sample_total,better,worse from samples;
 return response||jsonb_build_object('ruleset_version',version,'match_rank',jsonb_build_object(
  'rank',better+1,'total',sample_total,'percentile',case when sample_total>1 then round(100.0*worse/(sample_total-1),2) else 0 end,
  'result',current_match.result,'duration_ms',current_match.game_duration_ms,'mode',current_match.mode,
  'map_id',current_match.map_id,'difficulty',current_match.difficulty,'opponent',current_match.opponent,
  'order',case when current_match.result='win' then 'asc' else 'desc' end,'comparison','player-best-v1','ruleset_version',version));
end $$;
revoke all on function public.qqt_submit_result(jsonb) from public;
grant execute on function public.qqt_submit_result(jsonb) to anon,authenticated,service_role;

create or replace function qqt_private.ruleset_win_leaderboard_data(p_player_id uuid,p_ruleset_version text) returns jsonb
language sql stable security definer set search_path='' as $$
 with ranked as (
  select row_number() over(order by m.game_duration_ms,m.received_at,w.record_id) rank,w.record_id,w.nickname,w.victory_message,w.ip_display,
   m.game_duration_ms,m.player_id,m.client_match_id,m.client_total_ms,m.received_at
  from qqt_private.win_records w join qqt_private.match_results m using(client_match_id)
  where m.result='win' and m.ruleset_version=p_ruleset_version
 ), latest as (select client_match_id from ranked where player_id=p_player_id order by client_total_ms desc,received_at desc,client_match_id desc limit 1),
 selected as (
  select r.*,coalesce(r.player_id=p_player_id,false) is_mine,coalesce(r.client_match_id=(select client_match_id from latest),false) is_latest
  from ranked r where rank<=20 or r.client_match_id=(select client_match_id from latest)
 )
 select jsonb_build_object('leaderboard_contract_version',1,'ruleset_version',p_ruleset_version,'ruleset_contract_version',2,'rows',
  coalesce(jsonb_agg(jsonb_build_object('rank',rank,'record_id',record_id,'nickname',nickname,'game_duration_ms',game_duration_ms,
   'victory_message',victory_message,'player_ip',ip_display,'is_mine',is_mine,'is_latest',is_latest) order by rank),'[]'::jsonb)) from selected;
$$;
revoke all on function qqt_private.ruleset_win_leaderboard_data(uuid,text) from public,anon,authenticated,service_role;
-- Legacy endpoints remain v1-only, including for old cached browser clients.
create or replace function qqt_private.win_leaderboard_data(p_player_id uuid) returns jsonb
language sql stable security definer set search_path='' as $$
 select qqt_private.ruleset_win_leaderboard_data(p_player_id,'stock-window-v1')-'ruleset_version'-'ruleset_contract_version';
$$;

create or replace function public.qqt_my_win_leaderboard_ruleset(p_player_id uuid,p_player_secret text,p_ruleset_version text) returns jsonb
language plpgsql stable security definer set search_path='' as $$
declare player qqt_private.players;
begin
 if p_ruleset_version is null or p_ruleset_version not in ('stock-window-v1','bun-native-v2') or p_player_id is null or p_player_secret is null or p_player_secret !~ '^[0-9a-f]{64}$' then
  raise exception 'invalid identity or ruleset' using errcode='22023'; end if;
 select * into player from qqt_private.players where player_id=p_player_id;
 if not found then return qqt_private.ruleset_win_leaderboard_data(null,p_ruleset_version); end if;
 if player.secret_hash<>pg_catalog.sha256(convert_to(p_player_secret,'UTF8')) then raise exception 'identity credential mismatch' using errcode='42501'; end if;
 return qqt_private.ruleset_win_leaderboard_data(p_player_id,p_ruleset_version);
end $$;
revoke all on function public.qqt_my_win_leaderboard_ruleset(uuid,text,text) from public,service_role;
grant execute on function public.qqt_my_win_leaderboard_ruleset(uuid,text,text) to anon,authenticated;

create or replace function public.qqt_ruleset_contract() returns jsonb
language sql stable set search_path='' as $$ select jsonb_build_object('ruleset_contract_version',2,'rulesets',jsonb_build_array('stock-window-v1','bun-native-v2'),'legacy_missing_version','stock-window-v1'); $$;
revoke all on function public.qqt_ruleset_contract() from public;
grant execute on function public.qqt_ruleset_contract() to anon,authenticated,service_role;
notify pgrst,'reload schema';
commit;
