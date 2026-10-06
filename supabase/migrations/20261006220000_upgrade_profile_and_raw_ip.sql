begin;

-- Apply after the two existing incremental migrations; never rerun the initial schema.
alter table qqt_private.players
  add column if not exists raw_ip inet,
  add column if not exists profile_saved boolean not null default false;
comment on column qqt_private.players.raw_ip is
  'Last submitted source IP; forwarded headers are accepted and may be spoofed. Private, never returned by public RPC.';

-- Retain the exact existing validator, payload fingerprint, ranking and idempotency.
do $$
begin
  if to_regprocedure('qqt_private.qqt_submit_result_ranked(jsonb)') is null then
    alter function public.qqt_submit_result(jsonb) set schema qqt_private;
    alter function qqt_private.qqt_submit_result(jsonb) rename to qqt_submit_result_ranked;
  end if;
end;
$$;
revoke all on function qqt_private.qqt_submit_result_ranked(jsonb) from public, anon, authenticated, service_role;

create or replace function public.qqt_submit_result(p_payload jsonb) returns jsonb
language plpgsql security definer set search_path = '' as $$
declare
  response jsonb; pid uuid; mid uuid; previous qqt_private.players;
begin
  -- Use the same identity lock as the core and profile writer to avoid stale queue overwrites.
  pid := (p_payload->>'player_id')::uuid;
  mid := (p_payload->>'client_match_id')::uuid;
  perform pg_advisory_xact_lock(hashtextextended(pid::text, 0));
  select * into previous from qqt_private.players where player_id = pid;
  response := qqt_private.qqt_submit_result_ranked(p_payload);
  if previous.profile_saved then
    update qqt_private.players set nickname = previous.nickname, victory_message = previous.victory_message
      where player_id = pid;
  end if;
  return response || jsonb_build_object('match_upgraded', exists (
    select 1 from qqt_private.level_events where player_id = pid and client_match_id = mid
  ));
end;
$$;
revoke all on function public.qqt_submit_result(jsonb) from public;
grant execute on function public.qqt_submit_result(jsonb) to anon, authenticated, service_role;

create or replace function public.qqt_update_profile(
  p_player_id uuid, p_player_secret text, p_client_match_id uuid,
  p_nickname text, p_victory_message text
) returns jsonb
language plpgsql security definer set search_path = '' as $$
declare nick text := btrim(p_nickname); message text := btrim(p_victory_message);
begin
  if p_player_id is null or p_client_match_id is null or p_player_secret is null or
    p_player_secret !~ '^[0-9a-f]{64}$' or nick is null or char_length(nick) not between 1 and 24 or
    message is null or char_length(message) > 80 or (nick || message) ~ '[[:cntrl:]<>]' then
    raise exception 'invalid profile' using errcode = '22023';
  end if;
  perform pg_advisory_xact_lock(hashtextextended(p_player_id::text, 0));
  if not exists(select 1 from qqt_private.players where player_id = p_player_id and
      secret_hash = pg_catalog.sha256(convert_to(p_player_secret, 'UTF8'))) then
    raise exception 'identity credential mismatch' using errcode = '42501';
  end if;
  if not exists(select 1 from qqt_private.level_events where player_id = p_player_id and client_match_id = p_client_match_id) then
    raise exception 'profile requires completed level upgrade' using errcode = '22023';
  end if;
  update qqt_private.players set nickname = nick,
    victory_message = case when message = '' then victory_message else message end,
    profile_saved = true where player_id = p_player_id;
  return (select jsonb_build_object('saved', true, 'nickname', nickname, 'victory_message', victory_message)
    from qqt_private.players where player_id = p_player_id);
end;
$$;
revoke all on function public.qqt_update_profile(uuid,text,uuid,text,text) from public;
grant execute on function public.qqt_update_profile(uuid,text,uuid,text,text) to anon, authenticated;

-- Retire the previous redacted-only writer. The new writer computes masking inside SQL.
revoke all on function public.qqt_record_player_ip(uuid,text,text) from public, anon, authenticated, service_role;
create or replace function public.qqt_record_player_ip(p_player_id uuid, p_raw_ip text) returns jsonb
language plpgsql security definer set search_path = '' as $$
declare address inet; masked text;
begin
  if p_player_id is null or p_raw_ip is null or char_length(p_raw_ip) > 45 or
    p_raw_ip !~ '^[0-9a-fA-F:.]+$' then
    raise exception 'invalid network metadata' using errcode = '22023';
  end if;
  begin address := p_raw_ip::inet;
  exception when others then raise exception 'invalid network metadata' using errcode = '22023'; end;
  if family(address) = 4 then
    masked := split_part(host(address), '.', 1) || '.*.*.' || split_part(host(address), '.', 4);
  else
    -- Expand with PostgreSQL bit operations: compressed and mapped IPv6 work identically.
    masked := to_hex(get_byte(inet_send(address), 4)::integer * 256 + get_byte(inet_send(address), 5)::integer)
      || ':*:*:' || to_hex(get_byte(inet_send(address), 18)::integer * 256 + get_byte(inet_send(address), 19)::integer);
  end if;
  update qqt_private.players set raw_ip = address, ip_display = masked,
    ip_hash = null, ip_recorded_at = clock_timestamp() where player_id = p_player_id;
  if not found then raise exception 'player not found' using errcode = '22023'; end if;
  return jsonb_build_object('recorded', true, 'ip_display', masked);
end;
$$;
comment on function public.qqt_record_player_ip(uuid,text) is
  'Service-role-only raw inet writer. Source may be spoofed; response contains only server-masked display.';
revoke all on function public.qqt_record_player_ip(uuid,text) from public, anon, authenticated;
grant execute on function public.qqt_record_player_ip(uuid,text) to service_role;
notify pgrst, 'reload schema';
commit;
