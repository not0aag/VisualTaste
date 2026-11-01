import { describe, expect, it } from 'vitest'

import type { ModelVersion } from '../src/api/types'
import {
  binCentres,
  intervalText,
  overlaps,
  rungChanges,
  rungOf,
  versionRow,
} from '../src/lib/labModel'

const version = (metrics: Record<string, unknown>): ModelVersion => ({
  version: 'r3_20261031_0450_152',
  rung: 'r3',
  embedding_key: 'siglip2_b16_224',
  params: {},
  trained_on_events: 152,
  metrics,
  is_active: true,
  created_at: '2026-10-31',
})

describe('versionRow', () => {
  it('formats what exists and dashes what does not', () => {
    const row = versionRow(version({ replay_auc: 0.61234, score_p95_ms: 3.21, lambda: 0.003 }))
    expect(row).toMatchObject({ auc: '0.612', p95: '3.2 ms', lambda: '0.0030', ece: '--', ndcg: '--' })
  })

  it('falls back to the validation log loss before any replay', () => {
    expect(versionRow(version({ validation_log_loss: 0.68 })).logLoss).toBe('0.680')
  })

  it('treats NaN and nulls as missing', () => {
    expect(versionRow(version({ replay_auc: Number.NaN, ece: null })).auc).toBe('--')
  })
})

describe('rung changes', () => {
  it('reads the rung out of a version name', () => {
    expect(rungOf('r3_20261031_0450_152')).toBe('r3')
    expect(rungOf('r0_bootstrap')).toBe('r0')
    expect(rungOf('sim_r4_1729')).toBe('r4')
    expect(rungOf('r3+r5_20261101_0100_200')).toBe('r3+r5')
  })

  it('marks rung changes, not every retrain', () => {
    expect(
      rungChanges([
        { index: 0, model_version: 'r0_bootstrap' },
        { index: 30, model_version: 'r3_a_40' },
        { index: 50, model_version: 'r3_b_60' },
        { index: 200, model_version: 'r4_c_220' },
      ]),
    ).toEqual([
      { index: 0, rung: 'r0' },
      { index: 30, rung: 'r3' },
      { index: 200, rung: 'r4' },
    ])
  })
})

describe('intervals and histograms', () => {
  const a = { successes: 5, trials: 12, mean: 0.43, low: 0.22, high: 0.65 }
  const b = { successes: 60, trials: 100, mean: 0.6, low: 0.52, high: 0.68 }

  it('describes an interval', () => {
    expect(intervalText(a)).toBe('0.43 (0.22-0.65, n = 12)')
  })

  it('detects overlap and its absence', () => {
    expect(overlaps(a, b)).toBe(true)
    expect(overlaps(a, { ...b, low: 0.7, high: 0.8 })).toBe(false)
  })

  it('centres histogram bins', () => {
    expect(binCentres([0, 0.5, 1])).toEqual([0.25, 0.75])
  })
})
