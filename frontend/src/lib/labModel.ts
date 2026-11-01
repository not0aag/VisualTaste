/**
 * The Model Lab's pure logic, separate from React so Vitest can check it.
 */

import type { Interval, ModelVersion, RungSwitch } from '../api/types'

/** One row of the per-version metrics table, numbers already formatted. */
export interface VersionRow {
  version: string
  rung: string
  labels: number
  active: boolean
  auc: string
  logLoss: string
  ndcg: string
  ece: string
  p95: string
  lambda: string
}

const fmt = (value: unknown, places = 3): string =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(places) : '--'

export function versionRow(v: ModelVersion): VersionRow {
  const m = v.metrics
  return {
    version: v.version,
    rung: v.rung,
    labels: v.trained_on_events,
    active: v.is_active,
    auc: fmt(m.replay_auc),
    logLoss: fmt(m.replay_log_loss ?? m.validation_log_loss),
    ndcg: fmt(m.replay_ndcg_at_10),
    ece: fmt(m.ece),
    p95: typeof m.score_p95_ms === 'number' ? `${m.score_p95_ms.toFixed(1)} ms` : '--',
    lambda: typeof m.lambda === 'number' ? m.lambda.toPrecision(2) : '--',
  }
}

/** The rung a version string names: "r3_20261031_0450_152" -> "r3", "r0_bootstrap" -> "r0". */
export function rungOf(modelVersion: string): string {
  if (modelVersion.startsWith('sim_')) return modelVersion.split('_')[1] ?? modelVersion
  return modelVersion.split('_')[0] ?? modelVersion
}

/**
 * Vertical marker lines where the serving rung changed. Consecutive versions of
 * the same rung (online retrains every 20 labels) are not a switch, so they are
 * merged: the chart marks R0 -> R3, not every retrain.
 */
export function rungChanges(switches: RungSwitch[]): { index: number; rung: string }[] {
  const out: { index: number; rung: string }[] = []
  for (const s of switches) {
    const rung = rungOf(s.model_version)
    if (out.at(-1)?.rung !== rung) out.push({ index: s.index, rung })
  }
  return out
}

/** "0.41 (0.32-0.50, n = 37)". */
export function intervalText(interval: Interval): string {
  return `${interval.mean.toFixed(2)} (${interval.low.toFixed(2)}-${interval.high.toFixed(2)}, n = ${interval.trials})`
}

/** Whether two intervals overlap: the honest summary of an exploration-yield comparison. */
export function overlaps(a: Interval, b: Interval): boolean {
  return a.low <= b.high && b.low <= a.high
}

/** Bin centres for a histogram given its edges. */
export function binCentres(edges: number[]): number[] {
  return edges.slice(0, -1).map((edge, i) => (edge + (edges[i + 1] ?? edge)) / 2)
}
