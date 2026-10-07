begin;

-- Incremental Round 4 only; requires the deployed 20261006 ranked/IP upgrade.
-- Never rerun the initial schema or replace the IP writer.
create table if not exists qqt_private.profile_receipts (
  client_match_id uuid primary key references qqt_private.match_results on delete cascade,
  player_id uuid not null references qqt_private.players on delete cascade,
  request_hash bytea not null check (octet_length(request_hash) = 32),
  applied boolean not null,
  created_at timestamptz not null default clock_timestamp()
);
alter table qqt_private.profile_receipts enable row level security;
revoke all on qqt_private.profile_receipts from public, anon, authenticated, service_role;

create or replace function public.qqt_submit_result(p_payload jsonb) returns jsonb
language plpgsql security definer set search_path = '' as $$
declare response jsonb; pid uuid; mid uuid; previous qqt_private.players;
begin
  pid := (p_payload->>'player_id')::uuid;
  mid := (p_payload->>'client_match_id')::uuid;
  perform pg_advisory_xact_lock(hashtextextended(pid::text, 0));
  select * into previous from qqt_private.players where player_id = pid;
  response := qqt_private.qqt_submit_result_ranked(p_payload);
  -- Registration is the first authenticated result, not the first level upgrade.
  -- Preserve any existing name and declaration, including pre-profile_saved players.
  if previous.player_id is not null then
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

-- Authenticated read of one's own registration; no UUID, secret, or IP in response.
create or replace function public.qqt_get_profile(p_player_id uuid, p_player_secret text) returns jsonb
language plpgsql security definer set search_path = '' as $$
declare player qqt_private.players;
begin
  if p_player_id is null or p_player_secret is null or p_player_secret !~ '^[0-9a-f]{64}$' then
    raise exception 'invalid identity' using errcode = '22023';
  end if;
  select * into player from qqt_private.players where player_id = p_player_id;
  if not found then return jsonb_build_object('profile_contract_version', 2, 'registered', false); end if;
  if player.secret_hash <> pg_catalog.sha256(convert_to(p_player_secret, 'UTF8')) then
    raise exception 'identity credential mismatch' using errcode = '42501';
  end if;
  return jsonb_build_object('profile_contract_version', 2, 'registered', true,
    'nickname', player.nickname, 'victory_message', player.victory_message);
end;
$$;
revoke all on function public.qqt_get_profile(uuid,text) from public, service_role;
grant execute on function public.qqt_get_profile(uuid,text) to anon, authenticated;

create or replace function public.qqt_update_profile(
  p_player_id uuid, p_player_secret text, p_client_match_id uuid,
  p_nickname text, p_victory_message text
) returns jsonb
language plpgsql security definer set search_path = '' as $$
declare
  player qqt_private.players; match qqt_private.match_results;
  receipt qqt_private.profile_receipts; fingerprint bytea;
  message text := btrim(p_victory_message); applied boolean; superseded boolean;
begin
  if p_player_id is null or p_client_match_id is null or p_player_secret is null or
    p_player_secret !~ '^[0-9a-f]{64}$' or message is null or char_length(message) > 80 or
    message ~ '[[:cntrl:]<>]' or (p_nickname is not null and
      (char_length(btrim(p_nickname)) not between 1 and 24 or p_nickname ~ '[[:cntrl:]<>]')) then
    raise exception 'invalid profile' using errcode = '22023';
  end if;
  perform pg_advisory_xact_lock(hashtextextended(p_player_id::text, 0));
  select * into player from qqt_private.players where player_id = p_player_id;
  if not found or player.secret_hash <> pg_catalog.sha256(convert_to(p_player_secret, 'UTF8')) then
    raise exception 'identity credential mismatch' using errcode = '42501';
  end if;
  select * into match from qqt_private.match_results
    where client_match_id = p_client_match_id and player_id = p_player_id;
  if not found then raise exception 'profile requires own completed match' using errcode = '22023'; end if;
  -- Null is the new client contract; a matching name keeps the old signature usable.
  if p_nickname is not null and btrim(p_nickname) <> player.nickname then
    raise exception 'nickname is immutable' using errcode = '22023';
  end if;
  fingerprint := pg_catalog.sha256(convert_to(jsonb_build_object(
    'nickname', player.nickname, 'victory_message', message)::text, 'UTF8'));
  select * into receipt from qqt_private.profile_receipts where client_match_id = p_client_match_id;
  if found then
    if receipt.player_id <> p_player_id or receipt.request_hash <> fingerprint then
      raise exception 'profile request already accepted' using errcode = '22023';
    end if;
  else
    -- A late old card may retry after a newer match's profile succeeded.
    -- Use the frozen cumulative game time so offline old results arriving late
    -- cannot roll the newer declaration back. Receipt time/UUID break ties.
    superseded := exists(select 1 from qqt_private.profile_receipts r
      join qqt_private.match_results m using(client_match_id)
      where r.player_id = p_player_id and r.applied and
        (m.client_total_ms, m.received_at, m.client_match_id) >
        (match.client_total_ms, match.received_at, match.client_match_id));
    applied := not superseded;
    if applied then
      update qqt_private.players set
        victory_message = case when message = '' then victory_message else message end,
        profile_saved = true where player_id = p_player_id;
    end if;
    insert into qqt_private.profile_receipts(client_match_id,player_id,request_hash,applied)
      values(p_client_match_id,p_player_id,fingerprint,applied);
  end if;
  select * into player from qqt_private.players where player_id = p_player_id;
  return jsonb_build_object('saved', true, 'profile_contract_version', 2,
    'client_match_id', p_client_match_id, 'nickname', player.nickname,
    'victory_message', player.victory_message,
    'superseded', exists(select 1 from qqt_private.profile_receipts r
      join qqt_private.match_results m using(client_match_id)
      where r.player_id = p_player_id and r.applied and
        (m.client_total_ms, m.received_at, m.client_match_id) >
        (match.client_total_ms, match.received_at, match.client_match_id)));
end;
$$;
revoke all on function public.qqt_update_profile(uuid,text,uuid,text,text) from public, service_role;
grant execute on function public.qqt_update_profile(uuid,text,uuid,text,text) to anon, authenticated;
notify pgrst, 'reload schema';
commit;
