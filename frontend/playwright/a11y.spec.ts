/**
 * An automated accessibility audit of every page (axe-core, WCAG 2.1 A and
 * AA rules). Serious and critical findings fail the build; moderate ones are
 * printed so they stay visible.
 */

import AxeBuilder from '@axe-core/playwright'
import { expect, test } from '@playwright/test'

const PAGES = ['/feed', '/history', '/profile', '/lab', '/map', '/review']

for (const path of PAGES) {
  test(`${path} has no serious accessibility violations`, async ({ page }) => {
    await page.goto(path)
    await page.waitForLoadState('networkidle')
    await page.waitForTimeout(1500)

    const results = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .analyze()

    const blocking = results.violations.filter((v) => ['serious', 'critical'].includes(v.impact ?? ''))
    for (const v of results.violations) {
      console.log(`${path} [${v.impact}] ${v.id}: ${v.help} (${v.nodes.length} nodes)`)
      for (const node of v.nodes.slice(0, 3)) console.log(`    ${node.target.join(' ')}`)
    }
    expect(blocking.map((v) => `${v.id} (${v.nodes.length})`)).toEqual([])
  })
}
