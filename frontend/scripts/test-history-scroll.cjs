// Browser regression against a built, running app; all API data is synthetic.
// Requires Playwright (installed locally or available through NODE_PATH).
// BASE_URL defaults to http://127.0.0.1:8002; BROWSER_CHANNEL=chrome can use
// an installed Chrome instead of Playwright's bundled Chromium.
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

const baseURL = process.env.BASE_URL || 'http://127.0.0.1:8002';
const sessions = Array.from({ length: 30 }, (_, i) => ({
  session_id: `scroll-test-${i + 1}`, repository_id: 'scroll-test',
  title: `历史对话 ${i + 1}`, status: 'completed', updated_at: '2026-10-07T00:00:00Z',
}));
function session(id) {
  return {
    session_id: id, repository_id: 'scroll-test', status: 'completed',
    created_at: '2026-10-07T00:00:00Z', updated_at: '2026-10-07T00:00:00Z',
    streaming_answer: '', streaming_turn_id: null, source_issue: null,
    pending_question: null, runtime_steps: [], evidence: [], candidates: [], facts: [],
    issue_draft: null, memory_proposal: null, memory_case_id: null,
    investigation_summary: '', selected_memory_context: { preferences: [], cases: [] },
    memory_organization: { enabled: false, status: 'idle', last_error: null },
    messages: [{ id: 'synthetic-answer', role: 'assistant', content: `已切换到 ${id}`,
      created_at: '2026-10-07T00:00:00Z', citations: [], action: 'reply' }],
    open_question: null, focus_candidate_id: null, last_search_fingerprint: null,
    live_search_status: 'not_run', live_search_message: null, model_calls: 0,
    retrieval_calls: 0, prompt_tokens: null, completion_tokens: null,
    cached_input_tokens: null, cache_reported_input_tokens: 0,
    last_elapsed_ms: null, last_error: null,
  };
}

async function fixture(context, count) {
  await context.route('**/*', async route => {
    const request = route.request(), url = new URL(request.url());
    if (url.origin !== new URL(baseURL).origin) return route.abort();
    if (!url.pathname.startsWith('/api/')) return route.continue();
    assert.equal(request.method(), 'GET', 'UI test must not write to the backend');
    const path = url.pathname;
    if (path.endsWith('/events')) return route.fulfill({ contentType: 'text/event-stream', body: ': test\n\n' });
    let json = [];
    if (path === '/api/chat/sessions') json = sessions.slice(0, count);
    else if (path.startsWith('/api/chat/sessions/')) json = session(path.split('/')[4]);
    else if (path === '/api/chat/repositories') json = [{ id: 'scroll-test', label: 'owner/repo', issue_count: 30, source: 'local', github_url: null }];
    else if (path === '/api/chat/memory-settings') json = { auto_capture: false };
    return route.fulfill({ json });
  });
}

async function check(browser, width, height, count = 30) {
  const mobile = width <= 640;
  const context = await browser.newContext({ viewport: { width, height }, hasTouch: mobile });
  await fixture(context, count);
  const page = await context.newPage();
  await page.goto(baseURL, { waitUntil: 'domcontentloaded' });
  if (mobile) await page.getByRole('button', { name: '历史对话', exact: true }).click();
  const panel = page.locator('#chat-history');
  if (count) await panel.locator('.chat-history-scroll button').last().waitFor({ state: 'attached' });
  else await panel.getByText('还没有对话。', { exact: true }).waitFor();
  const dimensions = await panel.evaluate(el => ({
    height: el.clientHeight, content: el.scrollHeight,
    list: el.querySelector('.chat-history-scroll').clientHeight,
    listContent: el.querySelector('.chat-history-scroll').scrollHeight,
    overflow: getComputedStyle(el).overflowY,
  }));
  const mainTop = await page.locator('.chat-main-area').evaluate(el => el.scrollTop);
  const scrollTarget = mobile ? panel : panel.locator('.chat-history-scroll');
  const box = await scrollTarget.boundingBox();
  assert.ok(box && box.height > 0, 'History scroll target must not collapse to zero');
  if (count) {
    // In compact mode, wheel over the controls, not only over the list.
    await page.mouse.move(box.x + box.width / 2, box.y + (mobile ? Math.min(40, box.height / 2) : box.height / 2));
    await page.mouse.wheel(0, 10000);
    await page.waitForFunction(selector => document.querySelector(selector).scrollTop > 0,
      mobile ? '#chat-history' : '.chat-history-scroll');
    assert.equal(await page.locator('.chat-main-area').evaluate(el => el.scrollTop), mainTop,
      'Scrolling history must not scroll the conversation');
    if (mobile) {
      assert.ok(dimensions.height <= height * 0.45 + 1, 'Compact history must remain a small panel');
      await panel.evaluate(el => { el.scrollTop = 0; });
      const cdp = await context.newCDPSession(page);
      const x = box.x + box.width / 2, bottom = box.y + box.height - 15, top = box.y + 15;
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y: bottom }] });
      for (let i = 1; i <= 8; i++) {
        await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x, y: bottom + (top - bottom) * i / 8 }] });
        await page.waitForTimeout(16);
      }
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
      await page.waitForFunction(() => document.querySelector('#chat-history').scrollTop > 0);
      await cdp.detach();
    }
    const last = panel.locator('.chat-history-scroll button').last();
    await last.focus(); // Keyboard focus must reveal offscreen sessions.
    await page.keyboard.press('Enter');
    await page.getByText(`已切换到 scroll-test-${count}`, { exact: true }).waitFor();
    if (mobile) assert.equal(await panel.isVisible(), false, 'Switching session should close the compact panel');
    await page.locator('.chat-composer textarea').fill('继续对话的测试文本');
    assert.equal(await page.locator('.chat-composer textarea').inputValue(), '继续对话的测试文本');
  }
  if (mobile) {
    if (!(await panel.isVisible())) await page.getByRole('button', { name: '历史对话', exact: true }).click();
    await panel.evaluate(el => { el.scrollTop = 0; });
    await panel.getByPlaceholder('标题或仓库名称').fill('不存在的对话');
    await panel.getByText('没有匹配的对话。', { exact: true }).waitFor();
    await panel.getByRole('button', { name: '更多 ＋', exact: true }).focus();
    await page.keyboard.press('Enter');
    await panel.getByRole('button', { name: '质量评估', exact: true }).focus();
    await page.keyboard.press('Escape');
    assert.equal(await panel.isVisible(), false, 'Escape should close history');
  }
  console.log(JSON.stringify({ width, height, count, dimensions, result: 'passed' }));
  await context.close();
}

(async () => {
  const browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_CHANNEL ? { channel: process.env.BROWSER_CHANNEL } : {}) });
  try {
    for (const [width, height] of [[390, 568], [390, 844], [600, 768], [640, 480], [480, 320], [1440, 900]])
      await check(browser, width, height);
    await check(browser, 390, 568, 0);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
