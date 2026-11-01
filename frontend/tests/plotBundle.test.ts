import { describe, expect, it } from 'vitest'

import { bundleFor } from '../src/components/Plot'

describe('bundleFor, the smallest Plotly build a chart needs', () => {
  it('uses the basic build for lines and bars', () => {
    expect(bundleFor([{ type: 'scatter' }, { type: 'bar' }] as Plotly.Data[])).toBe('basic')
  })

  it('uses the WebGL build when any trace is scattergl', () => {
    expect(bundleFor([{ type: 'scatter' }, { type: 'scattergl' }] as Plotly.Data[])).toBe('gl2d')
  })

  it('treats an untyped trace as a plain scatter', () => {
    expect(bundleFor([{ x: [1], y: [2] }] as Plotly.Data[])).toBe('basic')
  })
})
