begin;

do $$
begin
  if to_regclass('qqt_private.profile_receipts') is null or
     to_regprocedure('public.qqt_get_profile(uuid,text)') is null then
    raise exception 'requires deployed Round 4 profile migration';
  end if;
end;
$$;

create table if not exists qqt_private.win_records (
  client_match_id uuid primary key references qqt_private.match_results on delete cascade,
  record_id uuid not null unique default gen_random_uuid(),
  nickname text not null check (char_length(nickname) between 1 and 24 and nickname !~ '[[:cntrl:]<>]'),
  victory_message text check (char_length(victory_message) <= 80 and victory_message !~ '[[:cntrl:]<>]'),
  raw_ip inet,
  ip_display text,
  ip_recorded_at timestamptz,
  snapshot_version smallint not null check (snapshot_version in (0,1))
);
alter table qqt_private.win_records enable row level security;
revoke all on qqt_private.win_records from public, anon, authenticated, service_role;
comment on column qqt_private.win_records.raw_ip is
  'First service-role IP received for this match. Forwarded headers may be spoofed; not an identity credential. Never public.';

-- Old wins have no recoverable match-bound declaration or IP. Do not copy current profile/IP.
insert into qqt_private.win_records(client_match_id,nickname,snapshot_version)
  select m.client_match_id,p.nickname,0 from qqt_private.match_results m
  join qqt_private.players p using(player_id) where m.result='win'
  on conflict(client_match_id) do nothing;
create index if not exists match_results_win_order
  on qqt_private.match_results(game_duration_ms,received_at,client_match_id) where result='win';

-- Preserve the entire deployed Round 4 validators, fingerprint and receipt semantics.
do $$
begin
  if to_regprocedure('qqt_private.qqt_submit_result_profile_v2(jsonb)') is null then
    alter function public.qqt_submit_result(jsonb) set schema qqt_private;
    alter function qqt_private.qqt_submit_result(jsonb) rename to qqt_submit_result_profile_v2;
  end if;
  if to_regprocedure('qqt_private.qqt_update_profile_v2(uuid,text,uuid,text,text)') is null then
    alter function public.qqt_update_profile(uuid,text,uuid,text,text) set schema qqt_private;
    alter function qqt_private.qqt_update_profile(uuid,text,uuid,text,text) rename to qqt_update_profile_v2;
  end if;
end;
$$;
revoke all on function qqt_private.qqt_submit_result_profile_v2(jsonb),
  qqt_private.qqt_update_profile_v2(uuid,text,uuid,text,text) from public, anon, authenticated, service_role;

create or replace function public.qqt_submit_result(p_payload jsonb) returns jsonb
language plpgsql security definer set search_path='' as $$
declare response jsonb; mid uuid; pid uuid;
begin
  response:=qqt_private.qqt_submit_result_profile_v2(p_payload);
  mid:=(p_payload->>'client_match_id')::uuid; pid:=(p_payload->>'player_id')::uuid;
  insert into qqt_private.win_records(client_match_id,nickname,victory_message,snapshot_version)
    select m.client_match_id,p.nickname,
      case when btrim(p_payload->>'victory_message')='' then p.victory_message
        else btrim(p_payload->>'victory_message') end,1
    from qqt_private.match_results m join qqt_private.players p using(player_id)
    where m.client_match_id=mid and m.player_id=pid and m.result='win'
    on conflict(client_match_id) do nothing;
  return response;
end;
$$;
revoke all on function public.qqt_submit_result(jsonb) from public;
grant execute on function public.qqt_submit_result(jsonb) to anon, authenticated, service_role;

create or replace function public.qqt_update_profile(
  p_player_id uuid,p_player_secret text,p_client_match_id uuid,p_nickname text,p_victory_message text
) returns jsonb language plpgsql security definer set search_path='' as $$
declare response jsonb; accepted boolean;
begin
  perform pg_advisory_xact_lock(hashtextextended(p_player_id::text,0));
  accepted:=exists(select 1 from qqt_private.profile_receipts where client_match_id=p_client_match_id);
  response:=qqt_private.qqt_update_profile_v2(p_player_id,p_player_secret,p_client_match_id,p_nickname,p_victory_message);
  -- The first validated receipt freezes this match even when its global profile is superseded.
  if not accepted then
    update qqt_private.win_records w set victory_message=
      case when btrim(p_victory_message)='' then w.victory_message else btrim(p_victory_message) end
    where w.client_match_id=p_client_match_id and w.snapshot_version=1;
  end if;
  return response;
end;
$$;
revoke all on function public.qqt_update_profile(uuid,text,uuid,text,text) from public, service_role;
grant execute on function public.qqt_update_profile(uuid,text,uuid,text,text) to anon, authenticated;

create or replace function public.qqt_record_match_ip(p_player_id uuid,p_client_match_id uuid,p_raw_ip text)
returns jsonb language plpgsql security definer set search_path='' as $$
declare metadata jsonb; record qqt_private.win_records;
begin
  if p_player_id is null or p_client_match_id is null then
    raise exception 'invalid network metadata' using errcode='22023';
  end if;
  perform pg_advisory_xact_lock(hashtextextended(p_player_id::text,0));
  if not exists(select 1 from qqt_private.match_results
    where client_match_id=p_client_match_id and player_id=p_player_id) then
    raise exception 'network metadata requires own completed match' using errcode='22023';
  end if;
  -- Reuse the deployed strict inet parser/masking and retain private player metadata.
  metadata:=public.qqt_record_player_ip(p_player_id,p_raw_ip);
  update qqt_private.win_records set raw_ip=p_raw_ip::inet,
    ip_display=metadata->>'ip_display',ip_recorded_at=clock_timestamp()
    where client_match_id=p_client_match_id and snapshot_version=1 and raw_ip is null;
  select * into record from qqt_private.win_records where client_match_id=p_client_match_id;
  if not found or record.snapshot_version=0 then return jsonb_build_object('recorded',false); end if;
  return jsonb_build_object('recorded',true,'ip_display',record.ip_display);
end;
$$;
revoke all on function public.qqt_record_match_ip(uuid,uuid,text) from public, anon, authenticated;
grant execute on function public.qqt_record_match_ip(uuid,uuid,text) to service_role;

create or replace function qqt_private.win_leaderboard_data(p_player_id uuid) returns jsonb
language sql stable security definer set search_path='' as $$
  with ranked as (
    select row_number() over(order by m.game_duration_ms,m.received_at,w.record_id) as rank,
      w.record_id,w.nickname,w.victory_message,w.ip_display,m.game_duration_ms,
      m.player_id,m.client_match_id,m.client_total_ms,m.received_at
    from qqt_private.win_records w join qqt_private.match_results m using(client_match_id)
    where m.result='win'
  ), latest as (
    select client_match_id from ranked where player_id=p_player_id
    order by client_total_ms desc,received_at desc,client_match_id desc limit 1
  ), selected as (
    select r.*,coalesce(r.player_id=p_player_id,false) is_mine,
      coalesce(r.client_match_id=(select client_match_id from latest),false) is_latest
    from ranked r where rank<=20 or r.client_match_id=(select client_match_id from latest)
  )
  select jsonb_build_object('leaderboard_contract_version',1,'rows',
    coalesce(jsonb_agg(jsonb_build_object('rank',rank,'record_id',record_id,'nickname',nickname,
      'game_duration_ms',game_duration_ms,'victory_message',victory_message,'player_ip',ip_display,
      'is_mine',is_mine,'is_latest',is_latest) order by rank),'[]'::jsonb))
  from selected;
$$;
revoke all on function qqt_private.win_leaderboard_data(uuid) from public, anon, authenticated, service_role;

create or replace function public.qqt_win_leaderboard() returns jsonb
language sql stable security definer set search_path='' as $$
  select qqt_private.win_leaderboard_data(null);
$$;
revoke all on function public.qqt_win_leaderboard() from public, service_role;
grant execute on function public.qqt_win_leaderboard() to anon, authenticated;

create or replace function public.qqt_my_win_leaderboard(p_player_id uuid,p_player_secret text) returns jsonb
language plpgsql stable security definer set search_path='' as $$
declare player qqt_private.players;
begin
  if p_player_id is null or p_player_secret is null or p_player_secret !~ '^[0-9a-f]{64}$' then
    raise exception 'invalid identity' using errcode='22023';
  end if;
  select * into player from qqt_private.players where player_id=p_player_id;
  if not found then return qqt_private.win_leaderboard_data(null); end if;
  if player.secret_hash<>pg_catalog.sha256(convert_to(p_player_secret,'UTF8')) then
    raise exception 'identity credential mismatch' using errcode='42501';
  end if;
  return qqt_private.win_leaderboard_data(p_player_id);
end;
$$;
revoke all on function public.qqt_my_win_leaderboard(uuid,text) from public, service_role;
grant execute on function public.qqt_my_win_leaderboard(uuid,text) to anon, authenticated;
comment on function public.qqt_win_leaderboard() is
  'Win records Top20, duration ASC / received_at ASC / random public record_id ASC. No credentials or player identifiers.';
comment on function public.qqt_my_win_leaderboard(uuid,text) is
  'Verified capability: Top20 with private request-scoped mine/latest flags plus at most one latest win outside Top20. Latest uses cumulative time / receipt time / private match UUID.';

notify pgrst,'reload schema';
commit;
