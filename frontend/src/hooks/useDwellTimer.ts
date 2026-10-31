/**
 * How long a card was actually looked at.
 *
 * This measures attention, not elapsed time, and the difference is the whole
 * point. Three rules, from brief section 7.9:
 *
 * 1. **`performance.now()`, not `Date.now()`.** A monotonic clock. `Date.now()`
 *    jumps when the system clock is corrected or the laptop resumes from
 *    sleep, which would produce negative or absurd durations.
 *
 * 2. **Paused on blur and on visibility change.** Switching tabs or alt-tabbing
 *    away is not looking at the image. Without this, the longest dwell times
 *    in the log would belong to whichever image was on screen when Alen went
 *    to make tea.
 *
 * 3. **Capped at 30 seconds.** Beyond half a minute the number says nothing
 *    about interest. The cap is enforced here and again server-side, because a
 *    client is not to be trusted with an invariant.
 *
 * Also tracked: time to first interaction, which is the gap between the card
 * appearing and the first key or gesture. A fast decision is a different
 * signal from a slow one, and it is cheap to record now and analyse later.
 *
 * All of this is **logged and never used as a training label** (section 7.5).
 * Dwell correlates with interest, but also with distraction, a slow image
 * load, and having walked away. Phase 12 tests whether it predicts anything;
 * until then it is data, not a feature.
 */

import { useCallback, useEffect, useRef } from 'react'

export const MAX_DWELL_MS = 30_000

export interface DwellReading {
  /** Visible, focused milliseconds, capped at `MAX_DWELL_MS`. */
  dwellMs: number
  /** Milliseconds from the card appearing to the first interaction, if any. */
  timeToFirstInteractionMs: number | null
}

export function useDwellTimer(cardKey: string | null): {
  read: () => DwellReading
  noteInteraction: () => void
  reset: () => void
} {
  /** Accumulated visible time from previous visible stretches. */
  const accumulated = useRef(0)
  /** When the current visible stretch began, or null while hidden. */
  const visibleSince = useRef<number | null>(null)
  const firstInteraction = useRef<number | null>(null)

  const reset = useCallback(() => {
    accumulated.current = 0
    visibleSince.current = isVisible() ? performance.now() : null
    firstInteraction.current = null
  }, [])

  // A new card restarts the clock. Keyed on the card id rather than on mount,
  // because the Feed keeps one component and swaps the image inside it.
  useEffect(() => {
    reset()
  }, [cardKey, reset])

  useEffect(() => {
    const pause = () => {
      if (visibleSince.current !== null) {
        accumulated.current += performance.now() - visibleSince.current
        visibleSince.current = null
      }
    }
    const resume = () => {
      if (visibleSince.current === null) visibleSince.current = performance.now()
    }
    const onVisibilityChange = () => (isVisible() ? resume() : pause())

    document.addEventListener('visibilitychange', onVisibilityChange)
    window.addEventListener('blur', pause)
    window.addEventListener('focus', resume)
    return () => {
      document.removeEventListener('visibilitychange', onVisibilityChange)
      window.removeEventListener('blur', pause)
      window.removeEventListener('focus', resume)
    }
  }, [])

  const read = useCallback((): DwellReading => {
    const live =
      visibleSince.current === null ? 0 : performance.now() - visibleSince.current
    const total = accumulated.current + live
    return {
      dwellMs: Math.min(Math.round(total), MAX_DWELL_MS),
      timeToFirstInteractionMs:
        firstInteraction.current === null ? null : Math.round(firstInteraction.current),
    }
  }, [])

  const noteInteraction = useCallback(() => {
    if (firstInteraction.current !== null) return
    const live =
      visibleSince.current === null ? 0 : performance.now() - visibleSince.current
    firstInteraction.current = accumulated.current + live
  }, [])

  return { read, noteInteraction, reset }
}

function isVisible(): boolean {
  // `document.hidden` is undefined in some test environments; treating that as
  // visible is the right default -- a missing API should not zero every dwell.
  return typeof document === 'undefined' || document.hidden !== true
}
