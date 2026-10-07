'use strict';
const assert = require('assert'), fs = require('fs'), path = require('path');
const { PGlite } = require('@electric-sql/pglite');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const out = 'runs/qqt_win_records_20261008';
(async () => {
  const db = new PGlite(); let browser;
  try {
    await db.exec('create role anon; create role authenticated; create role service_role;');
    for (const file of ['supabase/migrations/20261005140000_leaderboard.sql',
      'supabase/migrations/20261005173000_leaderboard_ip_and_best_win.sql',
      'supabase/manual_upgrade_20261006.sql', 'supabase/manual_profile_update_20261007.sql'])
      await db.exec(fs.readFileSync(file, 'utf8'));
    browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
    const checks = [];
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      const page = await browser.newPage({ viewport }), external = [], errors = [];
      await page.route('**/*', route => {
        if (route.request().url().startsWith('file:')) return route.continue();
        external.push(route.request().url()); return route.abort();
      });
      page.on('pageerror', e => errors.push(e.message));
      await page.goto(`file://${path.resolve('docs/20261008-qqt-win-records-sql-upgrade.html')}`);
      const blocks = await page.evaluate(() => Object.fromEntries(['preflight', 'upgrade', 'verify', 'rollback'].map(id => [id, document.getElementById(id).textContent])));
      assert.equal(blocks.upgrade, fs.readFileSync('supabase/manual_win_records_20261008.sql', 'utf8'));
      assert.equal(blocks.rollback, fs.readFileSync('supabase/manual_profile_update_20261007.sql', 'utf8'));
      if (viewport.width === 1440) {
        const before = await db.exec(blocks.preflight);
        assert(Object.values(before[0].rows[0]).every(Boolean));
        await db.exec(blocks.upgrade);
        const validate = async () => {
          const result = await db.exec(blocks.verify);
          assert.equal(result[0].rows[0].contract_version, '1');
          assert.equal(result[1].rows[0].private_contract_version, '1');
          assert.equal(result[2].rows[0].fabricated_legacy_records, 0);
          assert.deepEqual(result[3].rows[0], { anon_table_read: false, anon_ip_write: false, service_ip_write: true, capability_read: true });
          assert.equal(result[4].rows[0].relrowsecurity, true);
        };
        await validate(); await db.exec(blocks.rollback);
        assert.equal((await db.query('select * from public.qqt_leaderboard()')).rows.length, 0);
        await db.exec(blocks.upgrade); await validate();
      }
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      assert.equal(await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme), 'light');
      assert.deepEqual(external, []); assert.deepEqual(errors, []);
      await page.screenshot({ path: `${out}/browser-local/${viewport.width}-sql-doc.png`, fullPage: true });
      await page.screenshot({ path: `${out}/browser-local/${viewport.width}-sql-doc-top.png` });
      checks.push({ viewport, embeddedSqlExact: true, noExternalRequests: true, noOverflow: true, errors });
      await page.close();
    }
    fs.writeFileSync(`${out}/html-checks.json`, JSON.stringify({ localPreflightUpgradeVerifyRollbackReupgrade: true, checks }, null, 2) + '\n');
    console.log('SQL document embedded bytes, executable preflight/upgrade/verify/rollback/reupgrade and Chromium desktop/mobile passed');
  } finally { if (browser) await browser.close(); await db.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
