/**
 * Tests for the dwell timer. The behaviours being protected are the three the
 * brief specifies: a monotonic clock, pausing when the window is not visible,
 * and a hard cap.
 */

import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { MAX_DWELL_MS, useDwellTimer } from '../src/hooks/useDwellTimer'

describe('useDwellTimer', () => {
  let now = 0

  beforeEach(() => {
    now = 1_000
    vi.spyOn(performance, 'now').mockImplementation(() => now)
    Object.defineProperty(document, 'hidden', { value: false, configurable: true })
  })

  const advance = (ms: number) => {
    now += ms
  }

  it('measures elapsed visible time', () => {
    const { result } = renderHook(() => useDwellTimer('card-1'))

    advance(2_500)

    expect(result.current.read().dwellMs).toBe(2_500)
  })

  it('caps the reading, so a forgotten tab is not a 40-minute dwell', () => {
    const { result } = renderHook(() => useDwellTimer('card-1'))

    advance(10 * 60 * 1000)

    expect(result.current.read().dwellMs).toBe(MAX_DWELL_MS)
  })

  it('stops counting while the window is blurred', () => {
    // Without this, the longest dwell in the log belongs to whichever image was
    // on screen when Alen went to make tea.
    const { result } = renderHook(() => useDwellTimer('card-1'))

    advance(1_000)
    act(() => {
      window.dispatchEvent(new Event('blur'))
    })
    advance(60_000)
    act(() => {
      window.dispatchEvent(new Event('focus'))
    })
    advance(500)

    expect(result.current.read().dwellMs).toBe(1_500)
  })

  it('stops counting while the tab is hidden', () => {
    const { result } = renderHook(() => useDwellTimer('card-1'))

    advance(800)
    act(() => {
      Object.defineProperty(document, 'hidden', { value: true, configurable: true })
      document.dispatchEvent(new Event('visibilitychange'))
    })
    advance(30_000)
    act(() => {
      Object.defineProperty(document, 'hidden', { value: false, configurable: true })
      document.dispatchEvent(new Event('visibilitychange'))
    })
    advance(200)

    expect(result.current.read().dwellMs).toBe(1_000)
  })

  it('restarts for a new card', () => {
    const { result, rerender } = renderHook(({ key }) => useDwellTimer(key), {
      initialProps: { key: 'card-1' },
    })
    advance(5_000)

    rerender({ key: 'card-2' })
    advance(300)

    expect(result.current.read().dwellMs).toBe(300)
  })

  it('records time to first interaction once, not on every key', () => {
    const { result } = renderHook(() => useDwellTimer('card-1'))

    advance(400)
    act(() => result.current.noteInteraction())
    advance(900)
    act(() => result.current.noteInteraction())

    expect(result.current.read().timeToFirstInteractionMs).toBe(400)
  })

  it('reports no interaction time when nothing was pressed', () => {
    const { result } = renderHook(() => useDwellTimer('card-1'))

    advance(600)

    expect(result.current.read().timeToFirstInteractionMs).toBeNull()
  })

  it('uses the monotonic clock rather than the wall clock', () => {
    // Date.now() jumps when the system clock is corrected or the laptop
    // resumes from sleep, which would produce negative or absurd durations.
    const dateSpy = vi.spyOn(Date, 'now')
    const { result } = renderHook(() => useDwellTimer('card-1'))

    advance(1_000)
    result.current.read()

    expect(dateSpy).not.toHaveBeenCalled()
  })
})
