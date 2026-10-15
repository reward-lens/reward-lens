// Run against an isolated WorkbenchServer serving the saved record and report.
// Use cached Playwright/axe through REWARD_LENS_PLAYWRIGHT_MODULE and REWARD_LENS_AXE_FILE.
// node browser-layout.mjs SERVER_HANDLE_JSON NEW_OUTPUT_DIR RECORD_JSON REPORT_HTML
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { pathToFileURL } from 'node:url';
const { chromium } = await import(process.env.REWARD_LENS_PLAYWRIGHT_MODULE
  ? pathToFileURL(process.env.REWARD_LENS_PLAYWRIGHT_MODULE).href : 'playwright');
async function verifyOffline([recordFile, reportFile, output, expectationsFile]) {
  if (!expectationsFile) throw new Error('offline usage: --offline RECORD_JSON REPORT_HTML NEW_OUTPUT_DIR EXPECTATIONS_JSON');
  const { isDeepStrictEqual } = await import('node:util');
  const recordBytes = await fs.readFile(recordFile);
  const reportBytes = await fs.readFile(reportFile);
  const record = JSON.parse(recordBytes);
  const expected = JSON.parse(await fs.readFile(expectationsFile, 'utf8'));
  const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
  const results = { state: 'RUNNING', expectedId: record.assay_id, recordSha256: hash(recordBytes), reportSha256: hash(reportBytes), views: [], errors: [], externalRequests: [], runtimeErrors: [] };
  const check = (condition, message) => { if (!condition) results.errors.push(message); };
  check(results.recordSha256 === expected.record_sha256 && results.reportSha256 === expected.report_sha256, 'offline exact input bytes');
  check(reportBytes.length === record.embedding.html_bytes, 'offline strict declared HTML byte budget');
  await fs.mkdir(output, { recursive: false });
  const browser = await chromium.launch({ headless: true });
  try {
    const context = await browser.newContext();
    const page = await context.newPage();
    page.on('request', request => { if (/^https?:/.test(request.url())) results.externalRequests.push(request.url()); });
    page.on('pageerror', error => results.runtimeErrors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') results.runtimeErrors.push(message.text()); });
    const axe = await fs.readFile(process.env.REWARD_LENS_AXE_FILE, 'utf8');
    for (const [width, height] of [[360, 800], [390, 844]]) {
      await page.setViewportSize({ width, height });
      await page.goto(pathToFileURL(reportFile).href, { waitUntil: 'networkidle' });
      await page.locator('#rl-report .report').waitFor();
      const embedded = JSON.parse(await page.locator('script#assay').textContent());
      check(isDeepStrictEqual(embedded, record), `${width} offline exact embedded record`);
      const data = await page.evaluate(() => ({ viewport: innerWidth, scrollWidth: document.documentElement.scrollWidth, text: document.body.innerText }));
      check(data.viewport === width && data.scrollWidth === width, `${width} offline document fits without clipping`);
      const join = page.locator('[data-section="join"]');
      const region = page.getByRole('region', { name: 'Join measurements', exact: true });
      const bounds = await join.boundingBox();
      check(bounds && bounds.x >= 0 && bounds.x + bounds.width <= width, `${width} join container fits`);
      const headings = await region.locator('thead th').allTextContents();
      check(isDeepStrictEqual(headings, expected.headings), `${width} preserves all five join column headings`);
      const rows = await region.locator('tbody tr').evaluateAll(nodes => nodes.map(row => Array.from(row.querySelectorAll('th,td')).map(cell => cell.textContent.trim())));
      check(isDeepStrictEqual(rows, expected.rows), `${width} preserves measured cells, absences, reasons and remedies`);
      check((await join.locator('[data-join-signal]').textContent()).trim() === expected.signal, `${width} preserves signal scope/refusal`);
      const links = await join.locator('a[data-evidence-entry]').evaluateAll(nodes => nodes.map(link => ({ entry: link.dataset.evidenceEntry, href: link.getAttribute('href'), resolved: !!document.getElementById(link.getAttribute('href').slice(1)) })));
      check(links.every(link => link.resolved), `${width} every evidence link resolves to its saved entry target`);
      await page.evaluate(axe);
      const accessibility = await page.evaluate(async () => { const r = await axe.run(document, { runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21aa'] } }); return { violations: r.violations, incomplete: r.incomplete }; });
      check(accessibility.violations.length === 0, `${width} offline axe violations: ` + accessibility.violations.map(v => v.id).join(','));
      for (let tab = 0; tab < 150 && !await region.evaluate(e => e === document.activeElement); tab++) await page.keyboard.press('Tab');
      const focused = await region.evaluate(e => ({ focused: e === document.activeElement, outline: getComputedStyle(e).outlineStyle, outlineWidth: getComputedStyle(e).outlineWidth, scrollWidth: e.scrollWidth, clientWidth: e.clientWidth, left: e.scrollLeft }));
      check(focused.focused && focused.outline !== 'none' && focused.outlineWidth !== '0px', `${width} join has visible Tab focus`);
      await page.screenshot({ path: path.join(output, `offline-${width}-join-left.png`), fullPage: false });
      const accessedColumns = [];
      for (let index = 0; index < headings.length; index++) {
        const header = region.locator('thead th').nth(index);
        for (let press = 0; press < 100; press++) {
          const box = await header.boundingBox();
          const viewport = await region.boundingBox();
          if (box.x + box.width <= viewport.x + viewport.width + 1) break;
          await page.keyboard.press('ArrowRight');
          await page.waitForTimeout(60);
        }
        const box = await header.boundingBox();
        const viewport = await region.boundingBox();
        const visible = box.x >= viewport.x - 1 && box.x + box.width <= viewport.x + viewport.width + 1;
        check(visible, `${width} keyboard exposes join column ${headings[index]}`);
        accessedColumns.push({ heading: headings[index], visible, box });
      }
      const after = await region.evaluate(e => ({ focused: e === document.activeElement, left: e.scrollLeft }));
      check(after.focused && (focused.scrollWidth <= focused.clientWidth || after.left > focused.left), `${width} keyboard scroll retains focus and reaches last column`);
      await page.screenshot({ path: path.join(output, `offline-${width}-join-right.png`), fullPage: false });
      await join.screenshot({ path: path.join(output, `offline-${width}-join-section.png`) });
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.screenshot({ path: path.join(output, `offline-${width}-viewport.png`), fullPage: false });
      results.views.push({ width, ...data, headings, rows, signal: expected.signal, links, bounds, accessibility, focused, after, accessedColumns });
    }
    check(results.externalRequests.length === 0 && results.runtimeErrors.length === 0, 'offline zero HTTP requests and runtime errors');
    results.state = results.errors.length ? 'FAILED' : 'PASSED';
  } catch (error) { results.state = 'FAILED'; results.errors.push(String(error)); }
  finally {
    await fs.writeFile(path.join(output, 'browser-results.json'), JSON.stringify(results, null, 2) + '\n');
    await browser.close();
  }
  console.log(JSON.stringify({ state: results.state, views: results.views.length, errors: results.errors }));
  return results.state === 'PASSED' ? 0 : 1;
}
if (process.argv[2] === '--offline') process.exit(await verifyOffline(process.argv.slice(3)));

const [handleFile, output, recordFile, reportFile] = process.argv.slice(2);
if (!reportFile) throw new Error('Supply server handle, new output directory, saved record, and report.');
const { url } = JSON.parse(await fs.readFile(handleFile, 'utf8'));
const recordBytes = await fs.readFile(recordFile);
const reportBytes = await fs.readFile(reportFile);
const expectedId = JSON.parse(recordBytes).assay_id;
const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const axe = await fs.readFile(process.env.REWARD_LENS_AXE_FILE, 'utf8');
await fs.mkdir(output, { recursive: false });
const results = { state: 'RUNNING', expectedId, origin: new URL(url).origin,
  recordSha256: sha256(recordBytes), reportSha256: sha256(reportBytes), errors: [], views: [], console: [] };
const check = (condition, message) => { if (!condition) results.errors.push(message); };
const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  page.on('pageerror', error => results.console.push(String(error)));
  page.on('console', message => { if (message.type() === 'error') results.console.push(message.text()); });
  const unauthenticated = await context.request.get(results.origin + '/api/record/demo');
  check(unauthenticated.status() === 401, 'record requires authentication');
  await page.goto(url, { waitUntil: 'networkidle' });
  for (const [apiPath, bytes] of [['/api/record/demo', recordBytes], ['/api/report/demo', reportBytes]]) {
    const response = await context.request.get(results.origin + apiPath);
    check(response.status() === 200 && (await response.body()).equals(bytes), apiPath + ' exact saved bytes');
  }
  const projection = {};
  for (const kind of ['compare', 'history', 'queue']) {
    const response = await context.request.get(results.origin + '/api/' + kind + (kind === 'queue' ? '?since=0' : '/demo'));
    check(response.status() === 200, kind + ' projection is available');
    projection[kind] = await response.json();
  }
  for (const [name, width, height] of [['desktop', 1440, 1000], ['mobile360', 360, 800], ['mobile390', 390, 844]]) {
    await page.setViewportSize({ width, height });
    for (const route of ['/', '/state/case-loaded', '/state/compare-scorecard', '/state/history-graph', '/state/queue-idle']) {
      await page.goto(results.origin + route, { waitUntil: 'networkidle' });
      const data = await page.evaluate(() => {
        const rectangle = element => { const r = element.getBoundingClientRect(); return { top: r.top, bottom: r.bottom, left: r.left, right: r.right }; };
        return { text: document.body.innerText, viewport: innerWidth, scrollWidth: document.documentElement.scrollWidth,
          mainCount: document.querySelectorAll('main').length,
          routeRects: Array.from(document.querySelectorAll('.workbench-route')).map(e => ({ inMain: !!e.closest('main'), ...rectangle(e) })),
          chartTables: Array.from(document.querySelectorAll('.chart-data')).map(e => ({ text: e.innerText, ...rectangle(e) })) };
      });
      const file = `${name}-${route.replaceAll('/', '_')}.png`;
      await page.screenshot({ path: path.join(output, file), fullPage: true });
      await page.screenshot({ path: path.join(output, file.replace('.png', '-viewport.png')), fullPage: false });
      await page.evaluate(axe);
      const accessibility = await page.evaluate(async () => { const r = await axe.run(document, {runOnly: {type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21aa']}}); return { violations: r.violations, incomplete: r.incomplete }; });
      const cell = { name, route, screenshot: file, ...data, accessibility };
      results.views.push(cell);
      const label = `${name} ${route}`;
      check(data.scrollWidth === width, label + ' fits viewport without clipping content');
      check(data.mainCount === 1, label + ' has one main landmark');
      check(accessibility.violations.length === 0, label + ' axe violations: ' + accessibility.violations.map(v => v.id).join(','));
      if (route.includes('compare') || route.includes('history') || route.includes('queue')) {
        const kind = route.includes('compare') ? 'compare' : route.includes('history') ? 'history' : 'queue';
        const reason = kind === 'compare' ? projection.compare.scorecard.reason : kind === 'history' ? projection.history.graph.reason : projection.queue.next_action;
        check(data.routeRects.length === 1 && data.routeRects[0].inMain, label + ' route content is inside main');
        const reasonNode = page.getByText(reason, { exact: true });
        const box = await reasonNode.boundingBox();
        cell.reason = { text: reason, box };
        check(box && box.y >= 0 && box.y + box.height <= height, label + ' actual reason is visible in first viewport');
        if (kind === 'compare') check(data.text.includes('NOT MEASURED') && data.text.includes(projection.compare.scorecard.remedy), label + ' preserves comparison refusal and remedy');
        if (kind === 'history') check(data.text.includes(projection.history.graph.remedy), label + ' preserves history remedy');
      } else {
        check(data.text.includes(JSON.parse(recordBytes).subject.version.id), label + ' displays the saved version');
        const chartTables = page.locator('.chart-data');
        for (let index = 0; index < await chartTables.count(); index++) await chartTables.nth(index).screenshot({ path: path.join(output, file.replace('.png', `-chart-${index}.png`)) });
        await page.evaluate(() => window.scrollTo(0, 0));
        const download = page.getByRole('button', { name: 'Download report', exact: true });
        const box = await download.boundingBox();
        check(box && box.y >= 0 && box.y + box.height <= height, label + ' download control visible');
        await page.keyboard.press('Tab');
        check(await download.evaluate(e => e === document.activeElement), label + ' download reachable by keyboard');
        const scroller = page.getByRole('region', { name: 'Declared measurements', exact: true });
        for (let tab = 0; tab < 40 && !await scroller.evaluate(e => e === document.activeElement); tab++) await page.keyboard.press('Tab');
        const focused = await scroller.evaluate(e => e === document.activeElement);
        const before = await scroller.evaluate(e => e.scrollLeft);
        await page.keyboard.press('ArrowRight');
        await page.waitForTimeout(150);
        const keyboard = await scroller.evaluate(e => ({ focused: e === document.activeElement, scrollLeft: e.scrollLeft, scrollWidth: e.scrollWidth, clientWidth: e.clientWidth, outline: getComputedStyle(e).outlineStyle, outlineWidth: getComputedStyle(e).outlineWidth }));
        cell.keyboard = keyboard;
        check(focused && keyboard.focused && keyboard.outline !== 'none' && keyboard.outlineWidth !== '0px', label + ' evidence scroller has visible keyboard focus');
        if (keyboard.scrollWidth > keyboard.clientWidth) check(keyboard.scrollLeft > before, label + ' right arrow reaches wide evidence columns');
        await page.screenshot({ path: path.join(output, file.replace('.png', '-keyboard.png')), fullPage: false });
      }
    }
  }
  await page.goto(results.origin, { waitUntil: 'networkidle' });
  await page.keyboard.press('Tab');
  const [download] = await Promise.all([page.waitForEvent('download'), page.keyboard.press('Enter')]);
  const downloadedFile = path.join(output, 'downloaded-report.html');
  await download.saveAs(downloadedFile);
  const downloadedBytes = await fs.readFile(downloadedFile);
  const embedded = /<script type="application\/json" id="assay">(.*?)<\/script>/s.exec(downloadedBytes.toString())?.[1];
  const exactEmbeddedRecord = !!embedded && JSON.stringify(JSON.parse(embedded)) === JSON.stringify(JSON.parse(recordBytes));
  check(downloadedBytes.equals(reportBytes), 'keyboard download delivers exact saved report bytes');
  check(exactEmbeddedRecord, 'downloaded report carries exact saved record');
  results.keyboardDownload = { suggestedFilename: download.suggestedFilename(), reportSha256: sha256(downloadedBytes), exactSavedReport: downloadedBytes.equals(reportBytes), exactEmbeddedRecord };
  check(results.console.length === 0, 'no browser runtime errors');
  results.state = results.errors.length ? 'FAILED' : 'PASSED';
} catch (error) {
  results.state = 'FAILED'; results.errors.push(String(error));
} finally {
  await fs.writeFile(path.join(output, 'browser-results.json'), JSON.stringify(results, null, 2) + '\n');
  await browser.close();
}
if (results.state !== 'PASSED') process.exitCode = 1;
console.log(JSON.stringify({ state: results.state, views: results.views.length, errors: results.errors }));
