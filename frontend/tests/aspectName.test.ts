import { describe, expect, it } from 'vitest'

import { aspectName } from '../src/components/SwipeCard'

describe('aspectName, the viewfinder readout of a frame', () => {
  it('names common landscape ratios', () => {
    expect(aspectName(1600, 1067)).toBe('3:2')
    expect(aspectName(1920, 1080)).toBe('16:9')
    expect(aspectName(1024, 768)).toBe('4:3')
  })

  it('turns the name round for portrait frames', () => {
    expect(aspectName(1067, 1600)).toBe('2:3')
    expect(aspectName(1080, 1920)).toBe('9:16')
  })

  it('calls a square a square', () => {
    expect(aspectName(800, 800)).toBe('1:1')
  })

  it('names nothing it is not close to', () => {
    expect(aspectName(1000, 880)).toBeNull() // 1.14: between 1:1 and 5:4
  })

  it('names nothing without dimensions', () => {
    expect(aspectName(null, 600)).toBeNull()
    expect(aspectName(800, 0)).toBeNull()
  })
})
