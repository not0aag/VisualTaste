import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { defineConfig, devices } from '@playwright/test'

/**
 * One smoke test, run against the real backend and a real browser.
 *
 * It exists to catch the class of failure that unit tests structurally cannot:
 * the keymap not reaching the handler, the optimistic update not advancing the
 * card, the event never arriving in SQLite. Each of those passes in isolation
 * and fails together.
 *
 * The servers are started by Playwright so the whole thing is one command, and
 * the backend is pointed at a disposable corpus under `.e2e` -- an automated
 * browser clicking through twenty swipes must never write into Alen's real
 * history.
 */

const here = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(here, '..')

// Built rather than hard-coded: a bare `python` resolves to whatever is first
// on PATH (a system 3.13 here, with none of the project's dependencies), and a
// hand-written relative path has to survive cmd.exe's own view of slashes.
const python = path.join(repoRoot, '.venv', 'Scripts', 'python.exe')

export default defineConfig({
  testDir: './playwright',
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: 'http://127.0.0.1:5174',
    trace: 'retain-on-failure',
    ...devices['Desktop Chrome'],
  },
  webServer: [
    {
      // The corpus is rebuilt as part of this command rather than in a
      // `globalSetup`, because Playwright starts its web servers *first*: the
      // setup hook was deleting and recreating the database while the API
      // already held a handle to the old file, so the feed saw no images at all.
      command:
        `"${python}" scripts/build_e2e_corpus.py --root ./.e2e --images 40 && ` +
        `"${python}" -m uvicorn visualtaste.api.main:app --host 127.0.0.1 --port 8001`,
      cwd: repoRoot,
      port: 8001,
      reuseExistingServer: false,
      timeout: 120_000,
      env: {
        // A sandboxed corpus, so an end-to-end run cannot disturb the real data.
        VT_PROJECT_ROOT: path.join(repoRoot, '.e2e'),
        VT_DATA_ROOT: path.join(repoRoot, '.e2e', 'data'),
        VT_ARCHIVE_ROOT: path.join(repoRoot, '.e2e', 'archive'),
        VT_MIN_FREE_GB: '0',
      },
    },
    {
      // A dedicated port, so an e2e run does not collide with a dev server.
      command: 'pnpm exec vite --port 5174 --strictPort',
      cwd: here,
      port: 5174,
      reuseExistingServer: false,
      timeout: 120_000,
      // Point the dev server's /api proxy at the sandboxed backend above,
      // rather than at the default port a real dev session uses.
      env: { VITE_API_TARGET: 'http://127.0.0.1:8001' },
    },
  ],
})
