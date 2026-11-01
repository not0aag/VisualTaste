import { describe, expect, it } from 'vitest'

import type { InterestCard } from '../src/api/types'
import {
  describeHistory,
  describeInterval,
  intervalGeometry,
  sortInterests,
} from '../src/lib/profileModel'

const interval = { likes: 14, dislikes: 8, mean: 0.625, low: 0.48, high: 0.754 }

describe('intervalGeometry', () => {
  it('places the interval and mean on a percent scale', () => {
    const g = intervalGeometry(interval)
    expect(g.left).toBeCloseTo(48)
    expect(g.width).toBeCloseTo(27.4)
    expect(g.mean).toBeCloseTo(62.5)
  })

  it('keeps a zero-width interval visible', () => {
    expect(intervalGeometry({ ...interval, low: 0.5, high: 0.5 }).width).toBe(0.5)
  })

  it('clamps to the bar', () => {
    const g = intervalGeometry({ ...interval, low: -0.1, high: 1.2 })
    expect([g.left, g.left + g.width]).toEqual([0, 100])
  })
})

describe('describeInterval', () => {
  it('says the numbers in words', () => {
    expect(describeInterval(interval)).toBe(
      '63% (90% interval 48-75%), from 14 likes and 8 dislikes nearby',
    )
  })

  it('uses the singular', () => {
    expect(describeInterval({ ...interval, likes: 1, dislikes: 1 })).toContain('1 like and 1 dislike')
  })
})

describe('describeHistory', () => {
  it('describes the first clustering, then births, retirements and quiet reclusters', () => {
    expect(
      describeHistory([
        { likes: 50, born: [0, 1], retired: [] },
        { likes: 75, born: [2], retired: [] },
        { likes: 175, born: [3], retired: [1] },
        { likes: 275, born: [], retired: [] },
      ]),
    ).toEqual([
      'At 50 likes: first 2 interests found.',
      'At 75 likes: new: #2.',
      'At 175 likes: new: #3; retired: #1.',
      'At 275 likes: reclustered; every interest kept its id.',
    ])
  })
})

describe('sortInterests', () => {
  it('puts the biggest first and breaks ties by id', () => {
    const card = (interest_id: number, n_likes: number): InterestCard => ({
      interest_id,
      n_likes,
      label: '',
      central_image_ids: [],
      like_rate: interval,
    })
    expect(sortInterests([card(2, 5), card(0, 9), card(1, 5)]).map((c) => c.interest_id)).toEqual([
      0, 1, 2,
    ])
  })
})
