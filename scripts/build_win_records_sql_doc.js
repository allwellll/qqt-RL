'use strict';
const fs=require('fs'),crypto=require('crypto'),assert=require('assert');
const upgrade=fs.readFileSync('supabase/manual_win_records_20261008.sql','utf8');
assert.equal(upgrade,fs.readFileSync('supabase/migrations/20261008090000_win_records.sql','utf8'));
const rollback=fs.readFileSync('supabase/manual_profile_update_20261007.sql','utf8');
const preflight=`select to_regclass('qqt_private.profile_receipts') as required_receipts,
  to_regprocedure('public.qqt_get_profile(uuid,text)') as required_profile,
  to_regprocedure('public.qqt_update_profile(uuid,text,uuid,text,text)') as required_writer,
  to_regprocedure('qqt_private.qqt_submit_result_ranked(jsonb)') as required_ranked_core;
select count(*) as historical_wins from qqt_private.match_results where result='win';
select relname,relrowsecurity from pg_class join pg_namespace n on n.oid=relnamespace
  where n.nspname='qqt_private' and relkind='r' order by relname;
`;
const verify=`select public.qqt_win_leaderboard()->>'leaderboard_contract_version' as contract_version;
select public.qqt_my_win_leaderboard(gen_random_uuid(),repeat('a',64))->>'leaderboard_contract_version' as private_contract_version;
select count(*) as records,count(*) filter(where snapshot_version=0) as legacy_records,
  count(*) filter(where snapshot_version=0 and (victory_message is not null or raw_ip is not null)) as fabricated_legacy_records
  from qqt_private.win_records;
select has_table_privilege('anon','qqt_private.win_records','SELECT') as anon_table_read,
  has_function_privilege('anon','public.qqt_record_match_ip(uuid,uuid,text)','EXECUTE') as anon_ip_write,
  has_function_privilege('service_role','public.qqt_record_match_ip(uuid,uuid,text)','EXECUTE') as service_ip_write,
  has_function_privilege('anon','public.qqt_my_win_leaderboard(uuid,text)','EXECUTE') as capability_read;
select relrowsecurity from pg_class where oid='qqt_private.win_records'::regclass;
`;
const escape=s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#x27;');
const sha=crypto.createHash('sha256').update(upgrade).digest('hex');
const block=(id,label,text)=>`<h2>${label}</h2><button type="button" data-copy="${id}">复制 SQL</button><pre id="${id}">${escape(text)}</pre>`;
const html=`<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QQ堂逐胜局排行榜 SQL 升级 · 20261008</title>
<style>html{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f7f9fb;color:#18252f;font:15px/1.65 system-ui,sans-serif}main{max-width:1040px;margin:auto;padding:24px}h1{font-size:26px;line-height:1.3}h2{font-size:19px;margin:28px 0 10px}p,li{overflow-wrap:anywhere}code,pre{font-family:ui-monospace,monospace;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;background:#eef3f6;border:1px solid #c8d5df;border-radius:6px;padding:16px}button{border:1px solid #8da5b4;background:#fff;color:#173f56;padding:7px 12px;border-radius:4px;cursor:pointer}strong{color:#194f69}.stop{border-left:4px solid #b84f58;padding-left:14px}ol{padding-left:22px}footer{border-top:1px solid #c8d5df;margin-top:28px;padding-top:12px;color:#52616c}@media(max-width:600px){main{padding:16px}h1{font-size:23px}pre{padding:12px}}</style></head><body><main>
<h1>QQ堂逐胜局排行榜 SQL 升级</h1><p>20261008 · 人工增量升级 · 当前正式站点仍使用已验收的 a3d7de9 版本。<strong>本次不执行正式 SQL、不 push、不发布前端或 Edge。</strong></p>
<h2>范围与数据边界</h2><p>前置条件为已部署的 20261006 排名/IP 增量和 20261007 Round 4 资料收据增量；禁止重跑初始 schema。新增私有 <code>win_records</code> 表和索引、两层保留旧验证器的赛果/资料包装、新逐局 IP writer、公开胜局 Top20 RPC 和 capability 本人榜单 RPC。旧资料读 RPC、旧公开榜单、旧 raw IP writer、结算玩家最佳排名合同均保留。</p>
<p>历史真实胜局各自出现一行，但历史逐局感言和 IP 明确为空，不复制当前玩家资料。历史昵称使用迁移时玩家昵称，无法重建更早改名史。新胜局首次确认后保存当局感言；第一份绑定 match 的资料收据可完成该局快照，重复请求不改写，空感言保留该局已有值。原始 IP 仅在私表，第一份 service-role match IP 快照不可被重试覆盖。转发头可伪造，IP 不用于认证，不承诺它是权威客户端地址。</p>
<p>榜单按本局胜利用时升序，再按服务端接收时间、随机公开 record_id 稳定排序，显示真实全局序号。record_id 是记录句柄，不是玩家 UUID 或 match UUID。公开读取不收 secret；本人窄 RPC 验证已有 UUID/secret 后仅返回布尔归属与最新标记。Top20 后最多追加一条榜外最新胜局。最新由已持久化胜局的累计游戏时长、接收时间和私有 match UUID 排序，不由昵称或浏览器猜测。</p>
<p>随机三字符后缀只是首次昵称输入的可编辑初值，使用无歧义大写字母和数字的安全随机源，允许碰撞。删除后缀、改名或清空后都不会自动补回；空昵称仍需用户自行填写后才能提交。首次实际输入保存后沿用既有不可改名合同。</p>
<h2>执行顺序</h2><ol><li>先保存项目备份和当前函数定义，确认本地验证提交；执行下面只读前置检查，必要项不能为 NULL，既有私表应启用 RLS。</li><li>人工执行完整升级 SQL 一次；它包含事务，可重入。不单独执行其中的 ALTER/INSERT。</li><li>执行升级后只读检查：两个合同版本均为 1，fabricated_legacy_records 为 0，匿名私表读取/IP写入为 false，service IP写入和capability读取为 true，RLS为 true。</li><li>回复当前 CX“SQL 已执行”。随后只读探测真实 PostgREST 新 RPC 合同；此探测不创建玩家或赛果。</li><li>确认发布授权后，部署本地新版 submit-result Edge，使来源 IP 绑定实际 match；再 push 前端、核验同 head Pages 和双视口。Edge 缺少新 RPC 时只回退到原私有玩家 IP writer，不伪造逐局 IP；新 Edge 尚未部署时新榜单 IP 可为空。</li></ol>
<p>文件：<code>supabase/migrations/20261008090000_win_records.sql</code> 与 <code>supabase/manual_win_records_20261008.sql</code> 逐字一致。SQL SHA256：<code>${sha}</code></p>
${block('preflight','执行前：只读核验',preflight)}
${block('upgrade','完整升级 SQL',upgrade)}
${block('verify','执行后：只读核验',verify)}
<h2>停止与回退条件</h2><div class="stop"><p>前置对象缺失、SQL 报错、合同版本不符、历史数据被填造、匿名权限扩大或 raw IP 出现在公开行时立即停止；不要发布前端、不要重跑初始 schema、不要通过正式写入“试一下”。事务提交前报错：执行 <code>ROLLBACK;</code>，查明原因后整体重试。</p><p>提交后如需功能回退，由用户人工执行下面已部署的 Round 4 资料包装恢复 SQL。它恢复旧赛果/资料公共函数，不删除新记录表、快照、私有副本或新读 RPC；旧正式前端/Edge继续可用。回退后不得发布新前端，需重新执行本次增量并验证。此为函数行为回退，不伪称撤销已提交的数据。</p></div>
<p>功能回退期间新增的胜局在重新升级时按历史记录补齐，感言与 IP 为空；不能用回退期间的玩家当前资料填造逐局快照。已有冻结快照保留不变。</p>
${block('rollback','提交后的兼容函数回退 SQL',rollback)}
<h2>PostgREST 只读合同</h2><p>公开接口 <code>/rest/v1/rpc/qqt_win_leaderboard</code> 的 POST body 为 <code>{}</code>，不包含身份；本人接口 <code>/rest/v1/rpc/qqt_my_win_leaderboard</code> body 含本地既有 <code>p_player_id</code> 和 <code>p_player_secret</code>，只用于该窄 RPC，不能贴到公开日志或分享。可用随机新 UUID 和合法格式的随机 secret 验证未注册分支，它只读且不会创建身份。</p><p>成功响应包含 <code>leaderboard_contract_version: 1</code> 和 <code>rows</code>；每行只有 rank、record_id、nickname、game_duration_ms、victory_message、player_ip、is_mine、is_latest。公开 RPC 的两个标记均为 false，绝不返回 player_id、client_match_id、secret、raw_ip 或 hash。</p>
<footer>文档为单文件浅色 HTML，无外部依赖。所有测试使用本地 PGlite 与浏览器 mock；正式数据库升级只能由用户人工执行。</footer></main>
<script>for(const b of document.querySelectorAll('[data-copy]'))b.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(document.getElementById(b.dataset.copy).textContent);b.textContent='已复制'}catch{b.textContent='请选中 SQL'}});</script></body></html>
`;
fs.writeFileSync('docs/20261008-qqt-win-records-sql-upgrade.html',html);
console.log(`Generated single-file SQL document; SQL SHA256 ${sha}`);
