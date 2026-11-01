/**
 * The smoke test the Phase 6 Definition of Done asks for: twenty keyboard
 * swipes and one undo, verified against the database rather than the UI.
 *
 * Why check SQLite directly: every unit test here can pass while the app is
 * broken end to end -- the keymap might not reach the handler, the optimistic
 * update might advance without posting, the event might never commit. Those
 * failures are invisible from inside the browser, and the only honest check is
 * to ask the database what it actually holds.
 */

import { execFileSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test, expect, type Page } from '@playwright/test'

// Absolute paths, resolved from this file: a relative one would be interpreted
// against whatever directory the runner happens to be in.
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
// The same venv python the config's web server uses; VT_PYTHON overrides it.
const PYTHON =
  process.env.VT_PYTHON ??
  (process.platform === 'win32'
    ? path.join(repoRoot, '.venv', 'Scripts', 'python.exe')
    : path.join(repoRoot, '.venv', 'bin', 'python'))
const DB_PATH = path.join(repoRoot, '.e2e', 'data', 'db', 'visualtaste.db')

/**
 * Query the sandboxed database through the project's own venv Python.
 *
 * Opened read-only so a mistake in a test's SQL cannot alter the state the
 * test is in the middle of asserting about.
 */
function queryDatabase(sql: string): string {
  const script = [
    'import sqlite3',
    `connection = sqlite3.connect("file:${DB_PATH.replace(/\\/g, '/')}?mode=ro", uri=True)`,
    `print(connection.execute(${JSON.stringify(sql)}).fetchone()[0])`,
  ].join('\n')

  return execFileSync(PYTHON, ['-c', script], { encoding: 'utf-8' }).trim()
}

async function waitForCard(page: Page) {
  await expect(page.getByTestId('swipe-card')).toBeVisible()
}

async function currentImageId(page: Page): Promise<string | null> {
  return page.getByTestId('swipe-card').getAttribute('data-image-id')
}

test.describe('the Feed, end to end', () => {
  // Each test starts on a fresh profile. The web server and its corpus are
  // shared across the whole run, so without this every test would inherit the
  // previous one's swipes -- and `profile/reset` is the product's own feature
  // for exactly this, archiving rather than deleting.
  test.beforeEach(async ({ request }) => {
    const response = await request.post('http://127.0.0.1:8001/api/profile/reset')
    expect(response.ok()).toBeTruthy()
  })

  test('twenty keyboard swipes and an undo reach the database', async ({ page }) => {
    await page.goto('/feed')
    await waitForCard(page)

    // A mix of all four verdicts, in the proportions a real session has.
    const keys = [
      'ArrowRight', 'ArrowLeft', 'ArrowDown', 's',
      'ArrowRight', 'ArrowRight', 'ArrowLeft', 'ArrowDown',
      'ArrowRight', 's', 'ArrowLeft', 'ArrowRight',
      'ArrowDown', 'ArrowRight', 'ArrowLeft', 's',
      'ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
    ]

    const seen: string[] = []
    for (const key of keys) {
      const before = await currentImageId(page)
      if (before) seen.push(before)

      await page.keyboard.press(key)

      // The card must actually change: an optimistic update that did not
      // advance would otherwise let every key land on the same image.
      await expect
        .poll(async () => currentImageId(page), { timeout: 10_000 })
        .not.toBe(before)
    }

    // Twenty distinct images, so no card was rated twice.
    expect(new Set(seen).size).toBe(20)

    // The counter in the header is the UI's own claim.
    await expect(page.getByTestId('label-count')).toContainText('20')

    // And this is the database's answer, which is the one that matters.
    await expect
      .poll(() => queryDatabase('SELECT COUNT(*) FROM events WHERE undone = 0 AND profile_id = (SELECT MAX(profile_id) FROM profiles)'), {
        timeout: 15_000,
      })
      .toBe('20')

    // All four actions were recorded, not just the first one.
    expect(
      queryDatabase('SELECT COUNT(DISTINCT action) FROM events WHERE undone = 0 AND profile_id = (SELECT MAX(profile_id) FROM profiles)'),
    ).toBe('4')

    // Every event is tied to the impression that offered it -- without that
    // link, propensity and exploration yield have nothing to measure.
    expect(
      queryDatabase('SELECT COUNT(*) FROM events WHERE impression_id IS NOT NULL AND profile_id = (SELECT MAX(profile_id) FROM profiles)'),
    ).toBe('20')

    // An impression exists for every card that was shown.
    const impressions = Number(queryDatabase('SELECT COUNT(*) FROM impressions WHERE profile_id = (SELECT MAX(profile_id) FROM profiles)'))
    expect(impressions).toBeGreaterThanOrEqual(20)

    // Dwell was recorded and respects the cap.
    expect(
      queryDatabase('SELECT COUNT(*) FROM events WHERE dwell_ms IS NOT NULL AND profile_id = (SELECT MAX(profile_id) FROM profiles)'),
    ).toBe('20')
    expect(
      queryDatabase('SELECT COUNT(*) FROM events WHERE dwell_ms > 30000 AND profile_id = (SELECT MAX(profile_id) FROM profiles)'),
    ).toBe('0')

    // Now undo the last one.
    await page.keyboard.press('z')

    await expect
      .poll(() => queryDatabase('SELECT COUNT(*) FROM events WHERE undone = 0 AND profile_id = (SELECT MAX(profile_id) FROM profiles)'), {
        timeout: 15_000,
      })
      .toBe('19')

    // Nothing was deleted: the undone row is still there, marked.
    expect(
      queryDatabase(
        'SELECT COUNT(*) FROM events WHERE profile_id = (SELECT MAX(profile_id) FROM profiles)',
      ),
    ).toBe('20')
    expect(
      queryDatabase(
        'SELECT COUNT(*) FROM events WHERE undone = 1 AND profile_id = (SELECT MAX(profile_id) FROM profiles)',
      ),
    ).toBe('1')

    await expect(page.getByTestId('label-count')).toContainText('19')
  })

  test('every card credits its source', async ({ page }) => {
    // Open Images is CC BY 2.0 and requires attribution, so this is a
    // licensing requirement rather than a presentational one.
    await page.goto('/feed')
    await waitForCard(page)

    for (let i = 0; i < 4; i += 1) {
      const attribution = page.getByTestId('attribution')
      await expect(attribution).toBeVisible()
      await expect(attribution).toContainText('Test Photographer')
      await expect(attribution.getByRole('link', { name: /CC0/ })).toBeVisible()

      const before = await currentImageId(page)
      await page.keyboard.press('ArrowDown')
      await expect.poll(async () => currentImageId(page)).not.toBe(before)
    }
  })

  test('the buttons do what the keys do', async ({ page }) => {
    await page.goto('/feed')
    await waitForCard(page)

    const before = await currentImageId(page)
    await page.getByTestId('btn-like').click()

    await expect.poll(async () => currentImageId(page)).not.toBe(before)
    await expect(page.getByTestId('label-count')).toContainText('1')
  })

  test('the explanation panel toggles with I', async ({ page }) => {
    await page.goto('/feed')
    await waitForCard(page)

    await expect(page.getByTestId('explanation-panel')).toBeHidden()

    await page.keyboard.press('i')
    await expect(page.getByTestId('explanation-panel')).toBeVisible()
    await expect(page.getByTestId('explanation-panel')).toContainText('cold_start')

    await page.keyboard.press('i')
    await expect(page.getByTestId('explanation-panel')).toBeHidden()
  })

  test('? lists every shortcut, and Escape closes it', async ({ page }) => {
    await page.goto('/feed')
    await waitForCard(page)

    await page.keyboard.press('?')
    await expect(page.getByTestId('shortcut-overlay')).toBeVisible()
    await expect(page.getByTestId('shortcut-overlay')).toContainText('Undo the last verdict')

    await page.keyboard.press('Escape')
    await expect(page.getByTestId('shortcut-overlay')).toBeHidden()
  })

  test('each verdict is announced to screen readers', async ({ page }) => {
    await page.goto('/feed')
    await waitForCard(page)

    await page.keyboard.press('ArrowLeft')
    await expect(page.getByTestId('announcer')).toHaveText('Disliked. Next image.')
    // Undo is offered once the verdict has been saved.
    await expect(page.getByTestId('btn-undo')).toBeEnabled()
    await page.keyboard.press('z')
    await expect(page.getByTestId('announcer')).toHaveText('Undone. The previous image is back.')
  })

  test('undo is offered only after a swipe', async ({ page }) => {
    await page.goto('/feed')
    await waitForCard(page)

    await expect(page.getByTestId('btn-undo')).toBeDisabled()

    await page.keyboard.press('ArrowRight')

    await expect(page.getByTestId('btn-undo')).toBeEnabled()
  })
})
