import { describe, expect, it } from 'vitest'

import type { FeedCard } from '../src/api/types'
import { nearestLiked } from '../src/components/SwipeCard'

const card = (explanation: Record<string, unknown>) => ({ explanation }) as unknown as FeedCard

describe('nearestLiked, the "close to images you liked" row', () => {
  it('reads the liked images R3 named', () => {
    const named = [
      { image_id: '00000000000000aa', similarity: 0.91 },
      { image_id: '00000000000000bb', similarity: 0.84 },
    ]
    expect(nearestLiked(card({ nearest_liked: named }))).toEqual(named)
  })

  it('is empty for rungs that name none (cold start, kNN, probes)', () => {
    expect(nearestLiked(card({}))).toEqual([])
    expect(nearestLiked(card({ nearest_liked: 'nope' }))).toEqual([])
  })

  it('drops malformed entries rather than rendering them', () => {
    const mixed = [{ image_id: '00000000000000aa', similarity: 0.9 }, { image_id: 7 }, null]
    expect(nearestLiked(card({ nearest_liked: mixed }))).toHaveLength(1)
  })
})
