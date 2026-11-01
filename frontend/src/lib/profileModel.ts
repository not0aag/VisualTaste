/**
 * The Profile page's logic, kept out of the component so Vitest can check it.
 *
 * Three small jobs: place a Beta interval on a 0-100% bar, describe each
 * recluster in a sentence, and order interest cards. Nothing here fetches.
 */

import type { InterestCard, InterestEvent, LikeRateInterval } from '../api/types'

/** Percent positions for drawing an interval: [left edge, width, mean marker]. */
export interface IntervalGeometry {
  left: number
  width: number
  mean: number
}

const clamp = (value: number) => Math.min(100, Math.max(0, value * 100))

export function intervalGeometry(interval: LikeRateInterval): IntervalGeometry {
  const left = clamp(interval.low)
  const right = clamp(interval.high)
  return { left, width: Math.max(right - left, 0.5), mean: clamp(interval.mean) }
}

/** "62% (90% interval 48-75%), from 14 likes and 8 dislikes nearby". */
export function describeInterval(interval: LikeRateInterval): string {
  const pct = (v: number) => `${Math.round(v * 100)}%`
  return (
    `${pct(interval.mean)} (90% interval ${Math.round(interval.low * 100)}-${pct(interval.high)}), ` +
    `from ${interval.likes} like${interval.likes === 1 ? '' : 's'} and ` +
    `${interval.dislikes} dislike${interval.dislikes === 1 ? '' : 's'} nearby`
  )
}

/** One line per recluster, oldest first, saying what changed. */
export function describeHistory(history: InterestEvent[]): string[] {
  return history.map((event, index) => {
    const parts: string[] = []
    if (index === 0) {
      parts.push(`first ${event.born.length} interests found`)
    } else {
      if (event.born.length) parts.push(`new: ${event.born.map((id) => `#${id}`).join(', ')}`)
      if (event.retired.length)
        parts.push(`retired: ${event.retired.map((id) => `#${id}`).join(', ')}`)
      if (!parts.length) parts.push('reclustered; every interest kept its id')
    }
    return `At ${event.likes} likes: ${parts.join('; ')}.`
  })
}

/** Biggest interests first; ties by id so the order is stable between visits. */
export function sortInterests(cards: InterestCard[]): InterestCard[] {
  return [...cards].sort((a, b) => b.n_likes - a.n_likes || a.interest_id - b.interest_id)
}
