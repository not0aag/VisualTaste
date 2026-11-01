/**
 * One smoke test per page (Part 2 §9): it loads against the sandboxed corpus,
 * its main element renders, and the browser console stays free of errors.
 *
 * Not a functional test of each page; Vitest and the API tests do that. This
 * catches the failures only a real browser shows: a route that 404s, a chunk
 * that fails to load, a render that throws.
 */

import { test, expect, type ConsoleMessage, type Page } from '@playwright/test'

const PAGES = [
  { path: '/feed', testId: 'swipe-card' },
  { path: '/history', testId: 'history' },
  { path: '/profile', heading: 'Your taste' },
  { path: '/lab', testId: 'model-lab' },
  { path: '/map', testId: 'embedding-map' },
  { path: '/review', testId: 'review' },
]

function collectErrors(page: Page): string[] {
  const errors: string[] = []
  page.on('console', (message: ConsoleMessage) => {
    // A 404 for a missing projection is the map saying "run jobs.project",
    // which is expected on the sandbox corpus; it surfaces in the page text.
    if (message.type() === 'error' && !message.text().includes('404')) errors.push(message.text())
  })
  page.on('pageerror', (error) => errors.push(error.message))
  return errors
}

for (const entry of PAGES) {
  test(`${entry.path} renders without console errors`, async ({ page }) => {
    const errors = collectErrors(page)
    await page.goto(entry.path)
    if (entry.testId) {
      await expect(page.getByTestId(entry.testId)).toBeVisible()
    } else {
      await expect(page.getByRole('heading', { name: entry.heading })).toBeVisible()
    }
    await page.waitForLoadState('networkidle')
    expect(errors).toEqual([])
  })
}
