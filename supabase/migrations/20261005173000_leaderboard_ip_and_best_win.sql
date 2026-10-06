begin;

-- Incremental migration for databases that already ran 20261005140000_leaderboard.sql.
-- The Edge Function stores only a redacted display value and a keyed digest; the raw
-- request IP never enters Postgres and is never returned by the leaderboard RPC.
alter table qqt_private.players
  add column if not exists ip_display text,
  add column if not exists ip_hash bytea;
alter table qqt_private.players
  drop constraint if exists players_ip_display_check;
alter table qqt_private.players
  add constraint players_ip_display_check check (
    ip_display is null or
    ip_display ~ '^[0-9]{1,3}\.\*\.\*\.[0-9]{1,3}$' or
    ip_display ~ '^[0-9a-fA-F]{1,4}:\*:\*:[0-9a-fA-F]{1,4}$'
  );
alter table qqt_private.players
  drop constraint if exists players_ip_hash_check;
alter table qqt_private.players
  add constraint players_ip_hash_check check (ip_hash is null or octet_length(ip_hash) = 32);

create index if not exists players_ip_hash_idx on qqt_private.players(ip_hash);

-- Reserved for a future trusted network-metadata writer. This entry point is
-- intentionally not executable by anon/authenticated; the current Edge Function does
-- not invoke it while forwarded-IP provenance is unverified.
create or replace function public.qqt_record_player_ip(
  p_player_id uuid, p_ip_display text, p_ip_hash_hex text
) returns jsonb
language plpgsql security definer set search_path = '' as $$
begin
  if p_player_id is null or p_ip_display is null or p_ip_hash_hex is null or
    p_ip_hash_hex !~ '^[0-9a-fA-F]{64}$' or
    (p_ip_display ~ '^[0-9]{1,3}\.\*\.\*\.[0-9]{1,3}$' and
      (split_part(p_ip_display, '.', 1)::integer > 255 or split_part(p_ip_display, '.', 4)::integer > 255)) or
    not (p_ip_display ~ '^[0-9]{1,3}\.\*\.\*\.[0-9]{1,3}$' or
      p_ip_display ~ '^[0-9a-fA-F]{1,4}:\*:\*:[0-9a-fA-F]{1,4}$') then
    raise exception 'invalid network metadata' using errcode = '22023';
  end if;
  update qqt_private.players
    set ip_display = p_ip_display, ip_hash = decode(p_ip_hash_hex, 'hex')
    where player_id = p_player_id;
  if not found then raise exception 'player not found' using errcode = '22023'; end if;
  return jsonb_build_object('recorded', true);
end;
$$;
comment on function public.qqt_record_player_ip(uuid,text,text) is
  'Reserved service-role-only network metadata writer; disabled until IP provenance is trusted.';

revoke all on function public.qqt_record_player_ip(uuid,text,text) from public, anon, authenticated;
grant execute on function public.qqt_record_player_ip(uuid,text,text) to service_role;
grant execute on function public.qqt_submit_result(jsonb) to service_role;

drop function if exists public.qqt_leaderboard();
create function public.qqt_leaderboard() returns table(
  rank bigint, nickname text, victory_message text, player_ip text,
  best_win_duration_ms bigint, level integer, progress integer,
  level_reached_ms bigint, last_level_up_ms bigint, wins integer, games integer, win_rate numeric)
language sql stable security definer set search_path = '' as $$
  with summary as (
    select s.*, p.nickname, p.victory_message, p.ip_display,
      (select min(m.game_duration_ms)::bigint from qqt_private.match_results m
       where m.player_id = s.player_id and m.result = 'win') as best_win_duration_ms
    from qqt_private.player_progress s join qqt_private.players p using(player_id)
    where s.games > 0
  ), ranked as (
    select row_number() over (order by level desc, level_reached_ms asc,
      best_win_duration_ms asc nulls last, wins desc,
      wins::numeric / nullif(games, 0) desc, player_id asc) as rank, summary.*
    from summary
  )
  select rank, nickname, victory_message, ip_display, best_win_duration_ms, level,
    points % 10, level_reached_ms, last_level_up_ms, wins, games,
    round(wins::numeric / nullif(games, 0), 4)
  from ranked order by rank limit 20;
$$;
comment on function public.qqt_leaderboard() is
  'Top 20 by level, level time, best real win duration, wins and win rate; IP is redacted.';
revoke all on function public.qqt_leaderboard() from public;
grant execute on function public.qqt_leaderboard() to anon, authenticated;

notify pgrst, 'reload schema';
commit;
