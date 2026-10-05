begin;

-- Only bounded SECURITY DEFINER RPCs are exposed to the publishable key.
create schema if not exists qqt_private;
revoke all on schema qqt_private from public, anon, authenticated;

create table qqt_private.players (
  player_id uuid primary key,
  secret_hash bytea not null check (octet_length(secret_hash) = 32),
  nickname text not null check (char_length(nickname) between 1 and 24 and nickname !~ '[[:cntrl:]<>]'),
  victory_message text not null check (char_length(victory_message) <= 80 and victory_message !~ '[[:cntrl:]<>]'),
  created_at timestamptz not null default clock_timestamp()
);
create table qqt_private.match_results (
  client_match_id uuid primary key,
  player_id uuid not null references qqt_private.players on delete cascade,
  result text not null check (result in ('win','loss','draw')),
  game_duration_ms integer not null check (game_duration_ms between 1000 and 240000),
  wall_duration_ms integer not null check (wall_duration_ms between 1000 and 3600000),
  client_total_ms bigint not null check (client_total_ms between 1000 and 1000000000000),
  opponent text not null check (char_length(opponent) between 1 and 120),
  difficulty text not null check (difficulty in ('easy','normal','hard','model','fixed')),
  seed bigint not null check (seed between 0 and 4294967295),
  mode text not null check (mode in ('1v1','1v2','2v2')),
  map_id text not null check (map_id in ('training806','arena')),
  completed_at timestamptz not null,
  received_at timestamptz not null default clock_timestamp(),
  client_version text not null check (client_version = 'dev' or client_version ~ '^[0-9a-f]{40}$'),
  payload_hash bytea not null,
  check (wall_duration_ms >= game_duration_ms * 0.75 and client_total_ms >= game_duration_ms)
);
create index match_results_player_received on qqt_private.match_results(player_id, received_at desc);
create table qqt_private.player_progress (
  player_id uuid primary key references qqt_private.players on delete cascade,
  points integer not null default 0 check (points >= 0),
  level integer not null default 1 check (level = 1 + points / 10),
  wins integer not null default 0 check (wins >= 0),
  games integer not null default 0 check (games >= wins),
  total_game_ms bigint not null default 0 check (total_game_ms >= 0),
  level_reached_ms bigint not null default 0 check (level_reached_ms between 0 and total_game_ms),
  last_level_up_ms bigint not null default 0 check (last_level_up_ms between 0 and level_reached_ms)
);
create table qqt_private.level_events (
  player_id uuid not null references qqt_private.players on delete cascade,
  level integer not null check (level >= 2),
  client_match_id uuid not null references qqt_private.match_results on delete cascade,
  cumulative_game_ms bigint not null check (cumulative_game_ms > 0),
  level_up_ms bigint not null check (level_up_ms > 0),
  created_at timestamptz not null default clock_timestamp(),
  primary key(player_id, level)
);
alter table qqt_private.players enable row level security;
alter table qqt_private.match_results enable row level security;
alter table qqt_private.player_progress enable row level security;
alter table qqt_private.level_events enable row level security;
-- No client policies: all direct base-table access is denied.
revoke all on all tables in schema qqt_private from public, anon, authenticated;

create function public.qqt_submit_result(p_payload jsonb) returns jsonb
language plpgsql security definer set search_path = '' as $$
declare
  pid uuid; mid uuid; secret text; fingerprint bytea; existing qqt_private.match_results;
  progress qqt_private.player_progress; old_level integer; old_reached bigint;
  nick text; message text; outcome text; game_ms integer; wall_ms integer;
  total_ms bigint; finished timestamptz; version text;
begin
  if p_payload is null or jsonb_typeof(p_payload) <> 'object' or
    (select count(*) from jsonb_object_keys(p_payload)) <> 16 or
    exists (select 1 from jsonb_object_keys(p_payload) k where k <> all(array[
      'player_id','client_match_id','player_secret','nickname','victory_message','result',
      'game_duration_ms','wall_duration_ms','client_total_ms','opponent','difficulty',
      'seed','mode','map_id','completed_at','client_version'])) or
    exists (select 1 from jsonb_each(p_payload) where value = 'null'::jsonb) then
    raise exception 'invalid payload' using errcode = '22023';
  end if;
  if exists (select 1 from jsonb_each(p_payload) where
    jsonb_typeof(value) <> case when key in ('game_duration_ms','wall_duration_ms','client_total_ms','seed')
      then 'number' else 'string' end) then
    raise exception 'invalid field types' using errcode = '22023';
  end if;
  pid := (p_payload->>'player_id')::uuid; mid := (p_payload->>'client_match_id')::uuid;
  secret := p_payload->>'player_secret'; nick := btrim(p_payload->>'nickname');
  message := btrim(p_payload->>'victory_message'); outcome := p_payload->>'result';
  game_ms := (p_payload->>'game_duration_ms')::integer;
  wall_ms := (p_payload->>'wall_duration_ms')::integer;
  total_ms := (p_payload->>'client_total_ms')::bigint;
  finished := (p_payload->>'completed_at')::timestamptz; version := p_payload->>'client_version';
  if secret !~ '^[0-9a-f]{64}$' or char_length(nick) not between 1 and 24 or
    char_length(message) > 80 or nick ~ '[[:cntrl:]<>]' or message ~ '[[:cntrl:]<>]' or
    outcome not in ('win','loss','draw') or game_ms not between 1000 and 240000 or
    wall_ms not between 1000 and 3600000 or wall_ms < game_ms * 0.75 or
    total_ms not between game_ms and 1000000000000 or
    p_payload->>'opponent' !~ '^(bun\.(coop_hunter|hunter|tactical_v2|random_roam)|model:[A-Za-z0-9_.-]{1,100})$' or
    p_payload->>'difficulty' not in ('easy','normal','hard','model','fixed') or
    p_payload->>'mode' not in ('1v1','1v2','2v2') or
    p_payload->>'map_id' not in ('training806','arena') or
    (p_payload->>'seed')::bigint not between 0 and 4294967295 or
    not isfinite(finished) or finished > clock_timestamp() + interval '5 minutes' or
    finished < clock_timestamp() - interval '7 days' or
    (version <> 'dev' and version !~ '^[0-9a-f]{40}$') then
    raise exception 'invalid result fields' using errcode = '22023';
  end if;
  fingerprint := pg_catalog.sha256(convert_to((p_payload - 'player_secret')::text, 'UTF8'));
  -- Serialize both identity claims and global match IDs, including concurrent retries.
  perform pg_advisory_xact_lock(hashtextextended(pid::text, 0));
  perform pg_advisory_xact_lock(hashtextextended(mid::text, 1));
  insert into qqt_private.players(player_id, secret_hash, nickname, victory_message)
    values(pid, pg_catalog.sha256(convert_to(secret,'UTF8')), nick, message) on conflict do nothing;
  if not exists(select 1 from qqt_private.players where player_id = pid and
    secret_hash = pg_catalog.sha256(convert_to(secret,'UTF8'))) then
    raise exception 'identity credential mismatch' using errcode = '42501';
  end if;
  select * into existing from qqt_private.match_results where client_match_id = mid;
  if found then
    if existing.player_id <> pid or existing.payload_hash <> fingerprint then
      raise exception 'match id already used' using errcode = '22023';
    end if;
  else
    if exists(select 1 from qqt_private.match_results where player_id = pid and
      received_at > clock_timestamp() - interval '10 seconds') or
      (select count(*) from qqt_private.match_results where player_id = pid and
        received_at > clock_timestamp() - interval '1 day') >= 60 then
      raise exception 'submission rate limit' using errcode = 'P0001';
    end if;
    insert into qqt_private.match_results values(mid, pid, outcome, game_ms, wall_ms, total_ms,
      p_payload->>'opponent', p_payload->>'difficulty', (p_payload->>'seed')::bigint,
      p_payload->>'mode', p_payload->>'map_id', finished, clock_timestamp(), version, fingerprint);
    insert into qqt_private.player_progress(player_id) values(pid) on conflict do nothing;
    select level, level_reached_ms into old_level, old_reached
      from qqt_private.player_progress where player_id = pid;
    update qqt_private.player_progress set
      points = points + case when outcome = 'win' then 3 else 1 end,
      level = 1 + (points + case when outcome = 'win' then 3 else 1 end) / 10,
      wins = wins + case when outcome = 'win' then 1 else 0 end,
      games = games + 1, total_game_ms = total_game_ms + game_ms
      where player_id = pid returning * into progress;
    if progress.level > old_level then
      update qqt_private.player_progress set level_reached_ms = total_game_ms,
        last_level_up_ms = total_game_ms - old_reached where player_id = pid returning * into progress;
      insert into qqt_private.level_events(player_id,level,client_match_id,cumulative_game_ms,level_up_ms)
        values(pid,progress.level,mid,progress.total_game_ms,progress.last_level_up_ms);
    end if;
    update qqt_private.players set nickname = nick, victory_message = message where player_id = pid;
  end if;
  select * into progress from qqt_private.player_progress where player_id = pid;
  return jsonb_build_object('level',progress.level,'points',progress.points,'wins',progress.wins,
    'games',progress.games,'total_game_ms',progress.total_game_ms,
    'level_reached_ms',progress.level_reached_ms,'last_level_up_ms',progress.last_level_up_ms);
end;
$$;
comment on function public.qqt_submit_result(jsonb) is
  'Untrusted client results; capability secret, bounded fields, per-player rate limits and idempotent match IDs. No client IP is accepted.';

create function public.qqt_leaderboard() returns table(
  rank bigint, nickname text, victory_message text, level integer, progress integer,
  level_reached_ms bigint, last_level_up_ms bigint, wins integer, games integer, win_rate numeric)
language sql stable security definer set search_path = '' as $$
  select row_number() over (order by s.level desc, s.level_reached_ms asc, s.wins desc,
    s.wins::numeric / s.games desc, p.player_id asc),
    p.nickname, p.victory_message, s.level, s.points % 10, s.level_reached_ms,
    s.last_level_up_ms, s.wins, s.games, round(s.wins::numeric / s.games, 4)
  from qqt_private.player_progress s join qqt_private.players p using(player_id)
  where s.games > 0 order by s.level desc, s.level_reached_ms asc, s.wins desc,
    s.wins::numeric / s.games desc, p.player_id asc limit 20;
$$;
revoke all on function public.qqt_submit_result(jsonb), public.qqt_leaderboard() from public;
grant execute on function public.qqt_submit_result(jsonb), public.qqt_leaderboard() to anon, authenticated;
notify pgrst, 'reload schema';
commit;
